import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
# autoreload

from data_helpers import *

# ignore all pandas warnings
import warnings
warnings.filterwarnings("ignore")

class simulator:
    def __init__(self,
                 boroughs = ['Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island'],
                 categories = ['Hazard', 'Illegal Tree Damage', 'Other', 'Prune', 'Remove Tree',  'Root/Sewer/Sidewalk'],
                 date_start = '2019-01-01',
                 date_end = '2019-12-31',
                 city_budget = False,
                 byborough_policy = True,
                 borough_budgets = [0.2,0.2,0.2,0.2,0.2],
                 inspection_policies = np.array([[0.1,0.1,0.1,0.1,0.2,0.4],
                                                    [0.1,0.1,0.1,0.1,0.2,0.4],
                                                    [0.1,0.1,0.1,0.1,0.2,0.4],
                                                    [0.1,0.1,0.1,0.1,0.2,0.4],
                                                    [0.1,0.1,0.1,0.1,0.2,0.4]]),
                 service_fractions = np.array([[0.9,0.9,0.9,0.9,0.9,0.9],
                                                    [0.9,0.9,0.9,0.9,0.9,0.9],
                                                    [0.9,0.9,0.9,0.9,0.9,0.9],
                                                    [0.9,0.9,0.9,0.9,0.9,0.9],
                                                    [0.9,0.9,0.9,0.9,0.9,0.9]]),
                random_seed = 311,
                recursive = 3,
                fcfs_violation = 0.99,
                only_consider_inspected = False,
                dropping_frequency = [28, 80, 26, 27, 36],
                drop_using_age = False,
                drop_age = None,
                save_logs = False,
                calc_multi_objectives = True,
                sla_objective = "delay_75",
                drop_cost = 100,
                equity = "max",
                save_label = '',
                save_path = './simulation_log/'):
        '''
        Initialize the simulator object.
        borough_budgets: 
            a list of numbers that sum up to 1, indicating the budget allocation for each borough
        inspection_policies: 
            a n_borough*n_category matrix, each row summing to 1, that determines the priority of each category in each borough
        service_fractions: 
            a n_borough*n_category matrix, each element within [0,1], that determines the probability that a service request is dropped or served
        '''
        
        # set the basic parameters
        self.boroughs = boroughs
        self.categories = categories
        self.date_start = date_start
        self.date_end = date_end
        self.fcfs_violation = fcfs_violation
        self.only_consider_inspected = only_consider_inspected
        self.dropping_frequency = np.array(dropping_frequency).round(0)
        self.calc_multi_objectives = calc_multi_objectives
        self.sla_objective = sla_objective
        self.drop_cost = drop_cost
        self.equity = equity

        # drop_all supercedes service_fractions. if drop_using_age is True, then all service requests above some drop_age are dropped at the dropping point
        self.drop_using_age = drop_using_age
        if self.drop_using_age:
            self.drop_age = drop_age
            if self.drop_age is None:
                self.drop_age = 100
        self.date_list = pd.date_range(start=date_start, end=date_end)
        self.recursive = recursive
        if self.recursive > 1:
            # repeat the date list for recursive times
            self.date_list = pd.to_datetime(np.tile(self.date_list, recursive))
        self.virtual_date_list = np.arange(len(self.date_list))
        np.random.seed(random_seed)

        self.city_budget = city_budget
        self.byborough_policy = byborough_policy
        self.save_label = save_label
        self.save_logs = save_logs
        self.save_path = save_path

        # set the policy parameters
        if self.city_budget:
            self.init_policy_citybudget(inspection_policies, service_fractions)
        else:
            self.init_policy(borough_budgets, inspection_policies, service_fractions)

        # initialize the data to self.inspections and self.srs
        self.init_data()

        # set the current day to the start date
        self.current_day = self.date_start

        # setup the dataframes to store the simulation results
        self.init_simulation_outcomes()
        self.reset()

    def reset(self):
        '''
        Reset the simulation to the start date
        '''
        self.current_day = self.date_start
        self.virtual_date = 0 # we use this to log the inspections, so when recursively using the data we'd still be able to calculate the delay
        self.init_simulation_outcomes()


    def init_data(self):
        if 'Other' in self.categories:
            mergedraw = read_raw_and_filter(merge_low_categories=True)
        else:
            mergedraw = read_raw_and_filter(merge_low_categories=False)
        
        mergedraw = mergedraw.query('SRCategory in @self.categories')
        mergedraw = mergedraw.query('BoroughCode in @self.boroughs')
        if self.only_consider_inspected:
            mergedraw = mergedraw.query('InspectionDate == InspectionDate')

        inspection_counts = filter_inspections(mergedraw, self.date_start, self.date_end).reset_index()
        srs = filter_srs(mergedraw, self.date_start, self.date_end)
        # preprocess the inspection counts for the virtual_date_list
        self.inspections = pd.DataFrame(columns = ['InspectionDate', 'InspectionCount'])
        self.inspections['InspectionDate'] = self.virtual_date_list
        self.inspections['InspectionCount'] = 0
        inspection_counts.rename(columns = {'InspectionCount': 'raw_count'}, inplace = True)
        for i in range(self.recursive):
            inspection_counts['virtual_inspection_date'] = pd.to_timedelta(inspection_counts.InspectionDate - pd.to_datetime(self.date_start)).dt.days + i*(pd.to_datetime(self.date_end) - pd.to_datetime(self.date_start)).days
            inspection_counts['virtual_inspection_date'] = inspection_counts['virtual_inspection_date'].astype(int)
            self.inspections = self.inspections.merge(inspection_counts[['virtual_inspection_date', 'raw_count']], left_on='InspectionDate', right_on='virtual_inspection_date', how='outer')
            self.inspections['InspectionCount'] = self.inspections['InspectionCount'].fillna(0)
            self.inspections['raw_count'] = self.inspections['raw_count'].fillna(0)
            self.inspections['InspectionCount'] = self.inspections['InspectionCount'] + self.inspections['raw_count']
            self.inspections['InspectionDate'] = self.inspections['InspectionDate'].fillna(self.inspections['virtual_inspection_date'])
            self.inspections = self.inspections.drop(columns = ['raw_count', 'virtual_inspection_date'])

        self.srs = srs
        self.virtual_date_list = np.arange(self.inspections.InspectionDate.max() + 1)
        self.mergedraw = mergedraw
        # if self.date_list is shorter than virtual date list, extend it with fake dates
        if len(self.virtual_date_list) > len(self.date_list):
            # turn date_list to an array, and then extend
            self.date_list = self.date_list.to_numpy()
            appendix = pd.date_range(start=self.date_end, periods = len(self.virtual_date_list) - len(self.date_list) + 1)[1:]
            self.date_list = np.concatenate([self.date_list, appendix])
            self.date_list = pd.to_datetime(self.date_list)
        
        self.inspections = self.inspections.set_index('InspectionDate')

    def init_policy(self, borough_budgets, inspection_policies, service_fractions):
        assert len(borough_budgets) == len(self.boroughs), "The number of borough budgets should be the same as the number of boroughs"
        # assert np.sum(borough_budgets) == 1, "The borough budgets should sum up to 1"
        assert inspection_policies.shape == (len(self.boroughs), len(self.categories)), "The inspection policy should be a n_borough*n_category matrix"
        # assert np.all(np.sum(inspection_policies, axis=1) == 1), "Each row of the inspection policy should sum up to 1"
        assert service_fractions.shape == (len(self.boroughs), len(self.categories)), "The service fractions should be a n_borough*n_category matrix"
        assert np.all(service_fractions >= 0) and np.all(service_fractions <= 1), "The service fractions should be within [0,1]"
        self.borough_budgets = borough_budgets
        
        self.policy = pd.DataFrame(list(itertools.product(self.boroughs, self.categories)), columns=['BoroughCode', 'SRCategory'])
        self.policy['policy'] = inspection_policies.flatten()
        self.policy['fraction'] = service_fractions.flatten()
    
    def init_policy_citybudget(self, inspection_policies, service_fractions):
        if self.byborough_policy:
            assert inspection_policies.shape == (len(self.boroughs), len(self.categories)), "The inspection policy should be a n_borough*n_category matrix"
            # assert np.all(np.sum(inspection_policies, axis=1) == 1), "Each row of the inspection policy should sum up to 1"
            assert service_fractions.shape == (len(self.boroughs), len(self.categories)), "The service fractions should be a n_borough*n_category matrix"
            self.policy = pd.DataFrame(list(itertools.product(self.boroughs, self.categories)), columns=['BoroughCode', 'SRCategory'])
            self.policy['policy'] = inspection_policies.flatten()
            self.policy['fraction'] = service_fractions.flatten()
        else:
            assert len(inspection_policies) == len(self.categories), "The inspection policy should be a n_category array"
            # assert np.sum(inspection_policies) == 1, "The inspection policy should sum up to 1"
            assert len(service_fractions) == len(self.categories), "The service fractions should be a n_category array"
            self.policy = pd.DataFrame(self.categories, columns=['SRCategory'])
            self.policy['policy'] = inspection_policies
            self.policy['fraction'] = service_fractions

    
    def init_simulation_outcomes(self):
        self.inspected_srs = pd.DataFrame(columns=['CreatedDate', 'BoroughCode', 'SRCategory', 'InspectionDate', 'GlobalID'])
        self.dropped_srs = pd.DataFrame(columns=['CreatedDate', 'BoroughCode', 'SRCategory', 'GlobalID'])
        self.backlog_srs = pd.DataFrame(columns=['CreatedDate', 'BoroughCode', 'SRCategory', 'GlobalID'])
    
    def simulate_day(self, date):
        # next get the number of inspections on this day, and determine which srs in the backlog should be inspected
        try:
            inspections_on_date = self.inspections.loc[self.virtual_date][0]
        except:
            self.virtual_date += 1
            return

        # determine the number of inspections in each borough: because we are assuming that boroughs do inspections on their own, we assume that the number of inspections allocated could be over the number of srs in the backlog
        try:
            inspections_in_boroughs = np.random.multinomial(inspections_on_date, self.borough_budgets)
        except:
            # print error message and return
            print(inspections_on_date)
            print('No inspections on this date:', date)
            print('borough_budgets:', self.borough_budgets)
            return 
        # print('inspections on date', date, ':', inspections_in_boroughs, flush=True)

        # get a copy of the backlog srs and policy
        backlog_srs = self.backlog_srs.copy(deep=True).set_index(['BoroughCode'])
        

        # within the boroughs, determine the number of inspections for each category,
        # and then determine which srs in the backlog should be inspected
        for borough in range(len(self.boroughs)):
            BoroughCode = self.boroughs[borough]
            backlog_srs = backlog_srs.reset_index().set_index(['BoroughCode'])
            backlog_srs['Inspected'] = 0 # initialize the inspected column to 0
            policy = self.policy.copy(deep=True).set_index(['BoroughCode'])
            inspections_in_borough = inspections_in_boroughs[borough]
            try:
                backlogs_in_borough = len(backlog_srs.loc[[BoroughCode]])
            except:
                continue
        
            if inspections_in_borough == 0:
                continue
            elif inspections_in_borough >= backlogs_in_borough:
                backlog_srs.loc[BoroughCode, 'Inspected'] = 1
            else: # the number of available inspections is less than the number of srs in the backlog, we need to determine the number of inspections for each category
                # get the number of srs in each category
                backlog_srs_in_borough = backlog_srs.loc[BoroughCode]
                try:
                    counts = backlog_srs_in_borough.groupby('SRCategory').Inspected.count().reset_index().rename(columns={'Inspected': 'count'})
                except:
                    print(backlog_srs_in_borough)
                counts = counts.merge(policy.loc[[BoroughCode]][['SRCategory', 'policy']], on=['SRCategory'])
                if counts['policy'].sum() != 0:
                    counts.policy = counts.policy / counts['policy'].sum()
                else:
                    counts.policy = 1 / len(counts)            

                # determine the number of inspections for each category: multinomial sampling but do not exceed the counts['count']
                try:
                    counts['Inspections_rand'] = np.random.multinomial(inspections_in_borough, counts['policy'])
                except:
                    print(counts)
                    print(backlog_srs_in_borough)
                    print(policy)
                counts['Inspections'] = np.minimum(counts['Inspections_rand'], counts['count'])
                total_excess = counts['Inspections_rand'].sum() - counts['Inspections'].sum()
                while total_excess > 0: 
                    # get the categories that have not been inspected enough
                    neg_excess_cats = counts[counts['Inspections'] < counts['count']]
                    # randomly choose one of them to add 1 to their inspections
                    weights = neg_excess_cats['policy']
                    weights = weights/np.sum(weights)
                    cat = np.random.choice(neg_excess_cats['SRCategory'], p=weights)
                    counts.loc[counts['SRCategory'] == cat, 'Inspections'] += 1
                    total_excess -= 1

                # update the backlog_srs for this borough&category only
                backlog_srs = backlog_srs.reset_index().set_index(['BoroughCode', 'SRCategory'])
                for i in range(len(counts)):
                    cur_inspection_count = counts['Inspections'][i]
                    if cur_inspection_count == 0:
                        continue
                    # backlog_srs.loc[(BoroughCode, counts['SRCategory'][i]), 'Inspected'][:counts['Inspections'][i]] = 1  
                    mask = (backlog_srs.index.get_level_values('BoroughCode') == BoroughCode) & (backlog_srs.index.get_level_values('SRCategory') == counts['SRCategory'][i])
                    cumulative_count = mask.cumsum()
                    cumulative_sum = cumulative_count[-1]

                    # determine the ones to inspect, considering violation to FCFS
                    if self.fcfs_violation == 0:
                        # if self.fcfs_violation == 0: no need to sample
                        backlog_srs.loc[(cumulative_count <= cur_inspection_count) & mask, 'Inspected'] = 1
                    else:
                        # need to sample from the cumulative_count
                        # extreme case: if self.fcfs_violation == 1: just sample from cur_inspection_count
                        sample_pool = int(np.round(cur_inspection_count + self.fcfs_violation * (cumulative_sum - cur_inspection_count)))
                        sample = np.random.choice(sample_pool, cur_inspection_count, replace=False) + 1 # need to +1 here because sample starts from 0, but we actually want the first to be 1, since that's when the first cumsum changes
                        backlog_srs.loc[np.isin(cumulative_count, sample) & mask, 'Inspected'] = 1
                        if backlog_srs.loc[np.isin(cumulative_count, sample) & mask, 'CreatedDate'].isna().sum() > 0:
                            print(backlog_srs.loc[np.isin(cumulative_count, sample) & mask, :])
                
            # update the backlog_srs and inspected_srs
            backlog_srs = backlog_srs.reset_index()

            inspected = backlog_srs[backlog_srs['Inspected'] == 1][['CreatedDate', 'BoroughCode', 'SRCategory', 'GlobalID']]
            inspected['InspectionDate'] = self.virtual_date
            if inspected.CreatedDate.isna().sum()>0:
                print(inspected)

            backlog_srs = backlog_srs[backlog_srs['Inspected'] == 0][['CreatedDate', 'BoroughCode', 'SRCategory', 'GlobalID']]

            # now do the dropping according to service fractions, do this for each borough separately, as they potentially have different cadences
            if self.virtual_date%self.dropping_frequency[borough] ==0:
                if backlog_srs[backlog_srs.BoroughCode == BoroughCode].shape[0] > 0:
                    if not self.drop_using_age:
                        backlog_srs = backlog_srs.merge(self.policy[['BoroughCode', 'SRCategory','fraction']], on=['BoroughCode', 'SRCategory'])
                        backlog_srs['drop'] = np.random.binomial(1, backlog_srs['fraction'])
                        dropped_on_date = backlog_srs[(backlog_srs['drop'] == 0) & (backlog_srs['BoroughCode'] == BoroughCode)]
                        self.dropped_srs = pd.concat([self.dropped_srs, dropped_on_date[['CreatedDate', 'BoroughCode', 'SRCategory', 'GlobalID']]])
                        backlog_srs = backlog_srs[(backlog_srs['drop'] == 1) | (backlog_srs['BoroughCode'] != BoroughCode)]
                    else:
                        backlog_srs['drop'] = np.where(backlog_srs['CreatedDate'] < self.virtual_date - self.drop_age, 0, 1)
                        dropped_on_date = backlog_srs[backlog_srs['drop'] == 0]
                        self.dropped_srs = pd.concat([self.dropped_srs, dropped_on_date[['CreatedDate', 'BoroughCode', 'SRCategory', 'GlobalID']]])
                        backlog_srs = backlog_srs[backlog_srs['drop'] == 1]

            self.backlog_srs = backlog_srs[['CreatedDate', 'BoroughCode', 'SRCategory', 'GlobalID']]

            self.inspected_srs = pd.concat([self.inspected_srs, inspected])

        # lastly grab the service requests that were created on this day
        self.get_srs_on_date(date)

        self.virtual_date += 1
    
    
    def simulate_day_citybudget(self, date):
        # first grab the service requests that were created on this day
        # we now have delayed dropping
        self.get_srs_on_date(date)
        
        # next get the number of inspections on this day, and determine which srs in the backlog should be inspected
        try:
            inspections_on_date = self.inspections.loc[self.virtual_date][0]
        except:
            self.virtual_date += 1
            print("No inspections on date")
            return

        # get a copy of the backlog srs and policy
        backlog_srs = self.backlog_srs.copy(deep=True).set_index(['BoroughCode'])
        backlog_srs['Inspected'] = 0 # initialize the inspected column to 0
        policy = self.policy.copy(deep=True)

        try:
            backlogs = len(backlog_srs)
        except:
            backlogs = -1

        if backlogs <= 0:
            return
        elif inspections_on_date >= backlogs:
            backlog_srs.loc['Inspected'] = 1
            # print('should be inspected on date', date, 'in borough', BoroughCode, ':', len(backlog_srs.loc[[BoroughCode]]), flush=True)
        else: # the number of available inspections is less than the number of srs in the backlog, we need to determine the number of inspections for each category
            # get the number of srs in each category
            if self.byborough_policy:
                counts = backlog_srs.groupby(['BoroughCode','SRCategory']).Inspected.count().reset_index().rename(columns={'Inspected': 'count'})
                counts = counts.merge(policy[['BoroughCode', 'SRCategory', 'policy']], on=['BoroughCode', 'SRCategory'])
            else:
                counts = backlog_srs.groupby(['SRCategory']).Inspected.count().reset_index().rename(columns={'Inspected': 'count'})
                counts = counts.merge(policy[['SRCategory', 'policy']], on=['SRCategory'])

            if counts['policy'].sum() != 0:
                counts.policy = counts.policy / counts['policy'].sum()
            else:
                counts.policy = 1 / len(counts)            

            # determine the number of inspections for each category: multinomial sampling but do not exceed the counts['count']
            try:
                counts['Inspections_rand'] = np.random.multinomial(inspections_on_date, counts['policy'])
            except:
                print(counts)
                print(backlog_srs)
                print(policy)
            counts['Inspections'] = np.minimum(counts['Inspections_rand'], counts['count'])
            total_excess = counts['Inspections_rand'].sum() - counts['Inspections'].sum()
            while total_excess > 0: 
                # get the rows that have not been inspected enough
                neg_excess_cats = counts[counts['Inspections'] < counts['count']].index
                # randomly choose one of them to add 1 to their inspections
                # random weights equal to the amount in backlog
                weights = counts.loc[neg_excess_cats, 'policy']
                weights = weights/np.sum(weights)
                try:
                    cat = np.random.choice(neg_excess_cats, p=weights)
                except Exception as e:
                    print(e)

                counts.loc[cat, 'Inspections'] += 1
                total_excess -= 1
            

    
            # update the backlog_srs
            backlog_srs = backlog_srs.reset_index().set_index(['BoroughCode', 'SRCategory'])
            for i in range(len(counts)):
                cur_inspection_count = counts['Inspections'][i]
                # if there is no inspection assigned we just skip
                if cur_inspection_count == 0:
                    continue
                
                # if there are assigned, we do the sampling and assign inspections to reports
                if self.byborough_policy:
                    mask = (backlog_srs.index.get_level_values('BoroughCode') == counts['BoroughCode'][i]) & (backlog_srs.index.get_level_values('SRCategory') == counts['SRCategory'][i])
                else:
                    mask = (backlog_srs.index.get_level_values('SRCategory') == counts['SRCategory'][i])

                cumulative_count = mask.cumsum()
                cumulative_sum = cumulative_count[-1]
                if self.fcfs_violation == 0:
                    # no need to sample
                    backlog_srs.loc[(cumulative_count <= counts['Inspections'][i]) & mask, 'Inspected'] = 1
                else:
                    # need to sample cur_inspection_count from the cumulative_count
                    sample_pool = int(np.round(cur_inspection_count + self.fcfs_violation * (cumulative_sum - cur_inspection_count)))
                    sample = np.random.choice(sample_pool, cur_inspection_count, replace=False) + 1 # need to +1 here because sample starts from 0, but we actually want the first to be 1, since that's when the first cumsum changes
                    backlog_srs.loc[np.isin(cumulative_count, sample) & mask, 'Inspected'] = 1

        # update the backlog_srs and inspected_srs
        backlog_srs = backlog_srs.reset_index()

        inspected = backlog_srs[backlog_srs['Inspected'] == 1][['CreatedDate', 'BoroughCode', 'SRCategory', 'GlobalID']]
        inspected['InspectionDate'] = self.virtual_date

        backlog_srs = backlog_srs[backlog_srs['Inspected'] == 0][['CreatedDate', 'BoroughCode', 'SRCategory', 'GlobalID']]

        # now do the dropping according to service fractions
        if self.virtual_date%39 ==0:
            if backlog_srs.shape[0] > 0:
                if not self.drop_using_age:
                    if self.byborough_policy:
                        backlog_srs = backlog_srs.merge(self.policy[['BoroughCode', 'SRCategory','fraction']], on=['BoroughCode', 'SRCategory'])
                        backlog_srs['drop'] = np.random.binomial(1, backlog_srs['fraction'])
                        dropped_on_date = backlog_srs[backlog_srs['drop'] == 0]
                        self.dropped_srs = pd.concat([self.dropped_srs, dropped_on_date[['CreatedDate', 'BoroughCode', 'SRCategory', 'GlobalID']]])
                        backlog_srs = backlog_srs[backlog_srs['drop'] == 1]
                    else:
                        backlog_srs = backlog_srs.merge(self.policy[['SRCategory','fraction']], on=['SRCategory'])
                        backlog_srs['drop'] = np.random.binomial(1, backlog_srs['fraction'])
                        dropped_on_date = backlog_srs[backlog_srs['drop'] == 0]
                        self.dropped_srs = pd.concat([self.dropped_srs, dropped_on_date[['CreatedDate', 'BoroughCode', 'SRCategory', 'GlobalID']]])
                        backlog_srs = backlog_srs[backlog_srs['drop'] == 1]
                else:
                    backlog_srs['drop'] = np.where(backlog_srs['CreatedDate'] < self.virtual_date - self.drop_age, 0, 1)
                    dropped_on_date = backlog_srs[backlog_srs['drop'] == 0]
                    self.dropped_srs = pd.concat([self.dropped_srs, dropped_on_date[['CreatedDate', 'BoroughCode', 'SRCategory', 'GlobalID']]])
                    backlog_srs = backlog_srs[backlog_srs['drop'] == 1]

        self.backlog_srs = backlog_srs[['CreatedDate', 'BoroughCode', 'SRCategory', 'GlobalID']]

        self.inspected_srs = pd.concat([self.inspected_srs, inspected])
        # return backlog_srs

       
        self.virtual_date += 1

    def get_srs_on_date(self, date):
        '''
        Get the service requests that were created on this day
        We now delay the dropping to when inspections are done
        '''
        try:
            srs_on_date = self.srs.loc[date].reset_index()
            srs_on_date['CreatedDate'] = self.virtual_date
            self.backlog_srs = pd.concat([self.backlog_srs, srs_on_date])
        except:
            pass
    
    def save_log_delay(self):
        ins = self.inspected_srs.copy(deep=True)
        srid_to_ct = pd.read_csv('./data_clean/srid_to_ct.csv')
        ins = ins.merge(srid_to_ct, on='GlobalID')
        ins = ins[ins.SRCategory.isin(['Hazard', 'Illegal Tree Damage', 'Other', 'Prune', 'Remove Tree',  'Root/Sewer/Sidewalk'])]
        ins['delay'] = ins['InspectionDate'] - ins['CreatedDate']
        self.log_delay_GEOID = ins.groupby(['SRCategory','BoroughCode', 'GEOID']).agg(
            delay_25 = pd.NamedAgg(column='delay', aggfunc=lambda x: x.quantile(0.25)),
            delay_median = pd.NamedAgg(column='delay', aggfunc=lambda x: x.quantile(0.50)),
            delay_75 = pd.NamedAgg(column='delay', aggfunc=lambda x: x.quantile(0.75)),
            delay_90 = pd.NamedAgg(column='delay', aggfunc=lambda x: x.quantile(0.90)),
            delay_95 = pd.NamedAgg(column='delay', aggfunc=lambda x: x.quantile(0.95)),
            delay_mean = pd.NamedAgg(column='delay', aggfunc='mean'),
                                                    ).reset_index()
        self.log_delay_GEOID.to_csv(self.save_path + 'log_delay_censustract_' + '_citybudget_' + str(self.city_budget) +  self.save_label + '.csv', index=False)

        self.log_delay_Borough = ins.groupby(['SRCategory','BoroughCode']).agg(
            delay_25 = pd.NamedAgg(column='delay', aggfunc=lambda x: x.quantile(0.25)),
            delay_median = pd.NamedAgg(column='delay', aggfunc=lambda x: x.quantile(0.50)),
            delay_75 = pd.NamedAgg(column='delay', aggfunc=lambda x: x.quantile(0.75)),
            delay_90 = pd.NamedAgg(column='delay', aggfunc=lambda x: x.quantile(0.90)),
            delay_95 = pd.NamedAgg(column='delay', aggfunc=lambda x: x.quantile(0.95)),
            delay_mean = pd.NamedAgg(column='delay', aggfunc='mean'),
                                                    ).reset_index()
        self.log_delay_Borough.to_csv(self.save_path + 'log_delay_borough_' + '_citybudget_' + str(self.city_budget) +  self.save_label + '.csv', index=False)


        # additional logging of percentage inspected
        gbins = self.inspected_srs.groupby(['SRCategory', 'BoroughCode']).size().reset_index(name='inspected')
        gbbacklog = self.backlog_srs.groupby(['SRCategory', 'BoroughCode']).size().reset_index(name='backlog')
        gbdrop = self.dropped_srs.groupby(['SRCategory', 'BoroughCode']).size().reset_index(name='dropped')
        gbtotal = pd.merge(gbins, gbbacklog, on=['SRCategory', 'BoroughCode'], how='outer')
        gbtotal = pd.merge(gbtotal, gbdrop, on=['SRCategory', 'BoroughCode'], how='outer')
        gbtotal = gbtotal.fillna(0)
        gbtotal['total'] = gbtotal['inspected'] + gbtotal['backlog'] + gbtotal['dropped']
        gbtotal['FracInspected'] = gbtotal['inspected'] / (gbtotal['total'] + 1)
        gbtotal = gbtotal.fillna(0)
        gbtotal = gbtotal[['BoroughCode', 'SRCategory', 'FracInspected']]
        self.gbtotal = gbtotal
        gbtotal.to_csv(self.save_path + 'log_fracinspected_borough_' + '_citybudget_' + str(self.city_budget) +  self.save_label + '.csv', index=False)

    def calc_objectives(self):
        # calculate sr counts and read in mean risk rating
        srcounts = self.srs.groupby(['BoroughCode', 'SRCategory']).size().reset_index(name='SRCount')
        risk = pd.read_csv('./data_clean/mean_risk_rating.csv')

        df_objectives = pd.merge(srcounts, risk, on=['BoroughCode', 'SRCategory'], how='outer')
        df_objectives = df_objectives.merge(self.gbtotal, on=['BoroughCode', 'SRCategory'], how='outer')
        df_objectives = df_objectives.merge(self.log_delay_Borough, on=['BoroughCode', 'SRCategory'], how='outer')
        df_objectives = df_objectives.fillna(0)

        # calculate the objectives
        df_objectives['delay_cost'] = df_objectives['SRCount'] * df_objectives['FracInspected'] * df_objectives[self.sla_objective] * df_objectives['RiskRating']
        df_objectives['drop_cost'] = df_objectives['SRCount'] * (1 - df_objectives['FracInspected']) * df_objectives['RiskRating'] * self.drop_cost
        df_objectives['total_cost'] = df_objectives['delay_cost'] + df_objectives['drop_cost']

        # save the objectives
        self.df_objectives = df_objectives.query('SRCategory != "Plant Tree"')
        self.efficiency_objectives = self.df_objectives['total_cost'].sum()
        if self.equity == "max":
            self.equity_objectives = df_objectives.groupby('BoroughCode')['total_cost'].sum().max()
        elif self.equity == "varfrac":
            def max_min(x):
                return np.max(x) - np.min(x)
            self.equity_objectives = df_objectives.groupby('SRCategory').apply(lambda x: max_min(x.FracInspected) * x.RiskRating.mean()).sum()
        elif self.equity == "varsla":
            def max_min(x):
                return np.max(x) - np.min(x)
            self.equity_objectives = df_objectives.groupby('SRCategory').apply(lambda x: max_min(x[self.sla_objective]) * x.RiskRating.mean()).sum()/100


    def simulate(self):
        '''
        Simulate the entire period
        '''
        for date in self.date_list:
            self.current_day = date
            if self.city_budget:
                self.simulate_day_citybudget(date)
            else:
                self.simulate_day(date)
        if self.save_logs:
            self.save_log_delay()
        if self.calc_multi_objectives:
            self.calc_objectives()
    