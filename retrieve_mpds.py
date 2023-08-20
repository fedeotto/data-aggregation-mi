"""
Script to retrieve datasets from MPDS using dedicated API
Visit https://developer.mpds.io/ for more information
"""

from mpds_client import MPDSDataRetrieval, MPDSDataTypes
import pandas as pd
import numpy as np
import pickle

props_list = [
            'seebeck coefficient',
            'electrical conductivity',
            'electrical resistivity',
            'thermal conductivity',
            'isothermal bulk modulus',
            'shear modulus',
            'band gap'
        ]

def query_mpds_data(api_key: str = ''):
    results = []
    client = MPDSDataRetrieval(api_key=api_key, verbose=True)
    client.dtype = MPDSDataTypes.PEER_REVIEWED

    fields={'P': [
        'sample.material.entry',
        'sample.material.chemical_formula',
        'sample.measurement[0].condition[0].name',
        'sample.measurement[0].condition[0].scalar',
        'sample.measurement[0].condition[0].units',
        'sample.measurement[0].property.name',
        'sample.measurement[0].property.scalar',
        'sample.measurement[0].property.units',
    ]}

    columns = ['Entry',
                'formula',
                'condition_name',
                'condition_scalar',
                'condition_unit',
                'property_name',
                'property_scalar',
                'property_unit']

    # query for all the data
    for prop in props_list:
        query = {"props": prop}
        result = client.get_dataframe(query,
                                        fields=fields,
                                        columns=columns
                                        )
        results.append(result) 
    with open('./datasets/raw_mpds_data.pkl', 'wb') as handle:
        pickle.dump(results, handle)


def preprocess_mpds_data():
    dataset_keys = {0 :'seebeck_te',
                    1 : 'sigma_te',
                    2 : 'rho_te',
                    3 : 'thermalcond_citrine',
                    4 : 'bulkmodulus_mp',
                    5 : 'shearmodulus_mp',
                    6 : 'bandgap_zhuo'}
    
    with open('./datasets/raw_mpds_data.pkl', 'rb') as handle:
        results = pickle.load(handle)
    
    for i, mpds_df in enumerate(results):   #discarding common entries
        non_mpds_data = pd.read_csv('./datasets/' + dataset_keys[i] + '.csv')
        common_formulas = set(mpds_df['formula']).intersection(set(non_mpds_data['formula']))
        mpds_df = mpds_df[~mpds_df['formula'].isin(common_formulas)]

        if dataset_keys[i] == 'bandgap_zhuo':
            gap_names = 'energy gap', 'energy gap for direct transition','energy gap for indirect transition'
            mpds_df = mpds_df[mpds_df['property_name'].isin(gap_names)]
            mpds_df = mpds_df[mpds_df['property_scalar'] > 0]

        mpds_df.to_csv('./datasets/' + dataset_keys[i] + '_mpds.csv', index=False)

if __name__ == '__main__':
    query_mpds_data(api_key='')
    preprocess_mpds_data()