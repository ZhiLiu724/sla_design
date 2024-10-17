import os
import torch
import argparse


tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
SMOKE_TEST = os.environ.get("SMOKE_TEST")

from bo_problem import *

from botorch.models.gp_regression import FixedNoiseGP, SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.sampling import draw_sobol_samples

import time
import warnings

from botorch import fit_gpytorch_mll
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import is_non_dominated

from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.utils.sampling import sample_simplex


if __name__ == "__main__":
    # parse args to read in batch size and n_steps
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_steps", type=int, default=10)
    parser.add_argument("--city_budget", type=int, default=1)
    parser.add_argument("--byborough_policy", type=int, default=1)
    parser.add_argument("--drop_by_age", type=int, default=0)
    parser.add_argument("--drop_age", type=int, default=100)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--drop_cost", type=int, default=100)
    parser.add_argument("--obj", type=str, default="delay_median")
    parser.add_argument("--equity", type=str, default="varsla") # equity is measured in either the max cost of borough or the weighted maximal difference in SLAs ("varsla")

    args = parser.parse_args()

    problem = parks_simulation(city_budget=args.city_budget,
                                byborough_policy=args.byborough_policy,
                                drop_by_age=args.drop_by_age,
                                drop_age=args.drop_age,
                                obj=args.obj,
                                equity=args.equity)


    BATCH_SIZE = args.batch_size
    N_STEPS = args.n_steps
    NUM_RESTARTS = 10 if not SMOKE_TEST else 2
    RAW_SAMPLES = 512 if not SMOKE_TEST else 4
    MC_SAMPLES = 128 if not SMOKE_TEST else 16


    def generate_initial_data(n=6):
        # generate training data
        train_x = draw_sobol_samples(bounds=problem.bounds, n=n, q=1).squeeze(1)
        # make train_x dtype torch.float64
        train_x = train_x.to(**tkwargs)
        train_obj = problem(problem, X = train_x)
        return train_x, train_obj


    def initialize_model(train_x, train_obj):
        # define models for objective and constraint
        train_x = normalize(train_x, problem.bounds)
        models = []
        for i in range(train_obj.shape[-1]):
            train_y = train_obj[..., i : i + 1]
            models.append(
                SingleTaskGP(
                    train_x, train_y
                )
            )
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        return mll, model


    standard_bounds = torch.zeros(2, problem.dim, **tkwargs)
    standard_bounds[1] = 1

    def optimize_qnehvi_and_get_observation(model, train_x, train_obj, sampler, ref_point):
        """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
        # partition non-dominated space into disjoint rectangles
        # first calculate a reference point based on the training data
        ref_point = train_obj.min(dim=0)[0] - 0.05
        acq_func = qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,  # use known reference point
            X_baseline=normalize(train_x, problem.bounds),
            prune_baseline=True,  # prune baseline points that have estimated zero probability of being Pareto optimal
            sampler=sampler,
        )
        # optimize
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=standard_bounds,
            q=BATCH_SIZE,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,  # used for intialization heuristic
            options={"batch_limit": 64, "maxiter": 2_000_000, "nonnegative": True}, # the batch_limit is the number of candidates considered in each batch
            sequential=True,
        )
        # observe new values
        new_x = unnormalize(candidates.detach(), bounds=problem.bounds)
        new_obj = problem(problem, new_x)
        return new_x, new_obj



    verbose = True

    hvs_qnehvi = []
    print(f"Running qNEHVI with {BATCH_SIZE} initial points on {problem.dim} objectives")

    if os.path.exists(f"./botorch_log/train_x_qnehvi_citybudget{args.city_budget}_boroughpolicy{args.byborough_policy}_dropbyage{args.drop_by_age}_dropcost{args.drop_cost}_objcol{args.obj}_equity{args.equity}.pt"):
        train_x_qnehvi = torch.load(f"./botorch_log/train_x_qnehvi_citybudget{args.city_budget}_boroughpolicy{args.byborough_policy}_dropbyage{args.drop_by_age}_dropcost{args.drop_cost}_objcol{args.obj}_equity{args.equity}.pt")
        train_obj_qnehvi = torch.load(f"./botorch_log/train_obj_qnehvi_citybudget{args.city_budget}_boroughpolicy{args.byborough_policy}_dropbyage{args.drop_by_age}_dropcost{args.drop_cost}_objcol{args.obj}_equity{args.equity}.pt")
    else:
        print(f"Getting initial sample of {BATCH_SIZE}...", flush=True)
        # call helper functions to generate initial training data and initialize model
        train_x_qnehvi, train_obj_qnehvi = generate_initial_data(
            n=BATCH_SIZE
        )

    mll_qnehvi, model_qnehvi = initialize_model(train_x_qnehvi, train_obj_qnehvi)

    # compute hypervolume
    ref_point = train_obj_qnehvi.min(dim=0)[0] - 0.05
    bd = DominatedPartitioning(ref_point=ref_point, Y=train_obj_qnehvi)
    volume = bd.compute_hypervolume().item()
    hvs_qnehvi.append(volume)

    # run N_BATCH rounds of BayesOpt after the initial random batch
    for iteration in range(1, N_STEPS+1):

        # check if the current evaluations have exceeded the number needed
        if train_obj_qnehvi.shape[0] >= BATCH_SIZE*N_STEPS:
            break

        t0 = time.monotonic()

        # fit the models
        try:
            fit_gpytorch_mll(mll_qnehvi)
        except:
            continue

        print(f"\nModel {iteration} fitted", flush=True)

        # define the qEI and qNEI acquisition modules using a QMC sampler
        qnehvi_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
        print(f"Sampler {iteration} defined")

        print(f"Optimizing {iteration}...", flush=True)

        # optimize acquisition functions and get new observations
        (
            new_x_qnehvi,
            new_obj_qnehvi
        ) = optimize_qnehvi_and_get_observation(
            model_qnehvi, 
            train_x_qnehvi, 
            train_obj_qnehvi, 
            qnehvi_sampler, 
            ref_point=ref_point
        )
        print(f"\nOptimization {iteration} done", flush=True)

        train_x_qnehvi = torch.cat([train_x_qnehvi, new_x_qnehvi])
        train_obj_qnehvi = torch.cat([train_obj_qnehvi, new_obj_qnehvi])

        # save train_x_qnehvi, train_obj_qnehvi
        
        torch.save(train_x_qnehvi, f"./botorch_log/train_x_qnehvi_citybudget{args.city_budget}_boroughpolicy{args.byborough_policy}_dropbyage{args.drop_by_age}_dropcost{args.drop_cost}_objcol{args.obj}_equity{args.equity}.pt")
        torch.save(train_obj_qnehvi, f"./botorch_log/train_obj_qnehvi_citybudget{args.city_budget}_boroughpolicy{args.byborough_policy}_dropbyage{args.drop_by_age}_dropcost{args.drop_cost}_objcol{args.obj}_equity{args.equity}.pt")
        
        ref_point = train_obj_qnehvi.min(dim=0)[0] - 0.05
        best_solution = train_obj_qnehvi.max(dim=0)[0]
        bd = DominatedPartitioning(ref_point=ref_point, Y=train_obj_qnehvi)
        volume = bd.compute_hypervolume().item()
        hvs_qnehvi.append(volume)


        # reinitialize the models so they are ready for fitting on next iteration
        mll_qnehvi, model_qnehvi = initialize_model(train_x_qnehvi, train_obj_qnehvi)

        t1 = time.monotonic()

        if verbose:
            print(
                f"\nBatch {iteration:>2}: Hypervolume (qNEHVI) = "
                f"({hvs_qnehvi[-1]:>4.2f}), "
                f"\nRef point = {ref_point}, "
                f"\nBest solution = {best_solution}, "
                f"time = {t1-t0:>4.2f}.",
                end="",
            )
        else:
            print(".", end="")