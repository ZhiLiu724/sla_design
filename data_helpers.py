import pandas as pd
import numpy as np
import os

def read_raw_and_filter(sr_file = 'data_raw/Forestry_Service_Requests_20231201.csv',
                        ins_file = 'data_raw/Forestry_Inspections_20231201.csv',
                        merge_low_categories = True):
    '''
    Read raw data and filter out unneeded columns.
    '''
    if f'merged_data_merge{merge_low_categories}.csv' in os.listdir('data_clean'):
        insmerge = pd.read_csv(f'data_clean/merged_data_merge{merge_low_categories}.csv')
        insmerge.CreatedDate = pd.to_datetime(insmerge.CreatedDate).dt.date
        insmerge.InspectionDate = pd.to_datetime(insmerge.InspectionDate).dt.date
        return insmerge
    
    # read the service requests and filter the columns we need
    sr = pd.read_csv(sr_file)
    srdf = sr[['SRCategory', 'SRType', 'BoroughCode', 'ServiceRequestParentGlobalID', 'GlobalID', 'CreatedDate']].copy(deep = True)
    srdf['IncidentGlobalID'] = np.where(srdf['ServiceRequestParentGlobalID'].notnull(), srdf['ServiceRequestParentGlobalID'], srdf['GlobalID'])
    srdf['CreatedDate'] = pd.to_datetime(srdf['CreatedDate']).dt.date

    # read the inspections and filter the two columns we need 
    ins = pd.read_csv(ins_file)
    ins['year'] = pd.to_datetime(ins['InspectionDate'], errors='coerce').dt.year
    ins = ins.query('year >= 2010 & year <=2023')
    insdf = ins[['ServiceRequestGlobalID', 'InspectionDate']]
    insmerge = pd.merge(insdf, srdf, how = 'right', right_on = 'IncidentGlobalID', left_on = 'ServiceRequestGlobalID')
    insmerge['InspectionDate'] = pd.to_datetime(insmerge['InspectionDate']).dt.date
    if merge_low_categories:
        insmerge['SRCategory'] = np.where(insmerge['SRCategory'].isin(['Rescue/Preservation', 'Remove Stump', 'Remove Debris', 'Pest/Disease', 'Claims', 'Planting Space']), 'Other', insmerge['SRCategory'])
    
    insmerge.to_csv(f'data_clean/merged_data_merge{merge_low_categories}.csv', index = False)

    return insmerge


def filter_srs(merged_df,
                    date_start = '2019-01-01',
                    date_end = '2020-01-01') -> pd.DataFrame:
    '''
    Filter service requests by date range. return a dataframe with CreatedDate, SRCategory (and BoroughCode, if use_borough == True) as the index.
    '''
    date_start = pd.to_datetime(date_start).date()
    date_end = pd.to_datetime(date_end).date()
    merged_df = merged_df.query('CreatedDate >= @date_start & CreatedDate <= @date_end')
    sr = merged_df[['CreatedDate', 'SRCategory', 'BoroughCode', 'GlobalID']]
    sr.CreatedDate = pd.to_datetime(sr.CreatedDate)
    sr = sr.set_index('CreatedDate')
    sr = sr.sort_index()
    return sr

def filter_inspections(merged_df,
                    date_start = '2019-01-01',
                    date_end = '2020-01-01') -> pd.DataFrame:
    '''
    Filter inspections by date range. return a dataframe with InspectionDate as the index and a count column.
    '''
    date_start = pd.to_datetime(date_start).date()
    date_end = pd.to_datetime(date_end).date()
    merged_df = merged_df.query('InspectionDate >= @date_start & InspectionDate <= @date_end')
    inscount = merged_df.groupby(['InspectionDate']).GlobalID.count().reset_index()
    inscount['InspectionDate'] = pd.to_datetime(inscount['InspectionDate'])
    inscount.sort_values(by = 'InspectionDate', inplace = True)
    inscount.set_index('InspectionDate', inplace = True)
    inscount.rename(columns = {'GlobalID': 'InspectionCount'}, inplace = True)
    return inscount