"""

Simple notebook to preprocess data for shear modulus and bulk modulus.

"""

import joblib
import pandas as pd

# %% MPDS DATA

# taking various properties from mpds
mpds_properties = joblib.load('./datasets_new/mpds_properties.pkl')

# selecting mpds shear modulus and bulk modulus
mpds_shear_modulus = mpds_properties[-7]
mpds_bulk_modulus = mpds_properties[-6]

# dropping duplicates by keeping last
mpds_shear_modulus = mpds_shear_modulus.drop_duplicates(subset=['formula'],
                                                        keep='last'
                                                        ).reset_index(drop=True)

mpds_shear_modulus = mpds_shear_modulus[['formula', 'property_scalar']]
mpds_shear_modulus = mpds_shear_modulus.rename(columns={'property_scalar': 'target'})


mpds_bulk_modulus = mpds_bulk_modulus.drop_duplicates(subset=['formula'],
                                                      keep='last'
                                                      ).reset_index(drop=True)

mpds_bulk_modulus = mpds_bulk_modulus[['formula', 'property_scalar']]

mpds_bulk_modulus = mpds_bulk_modulus.rename(columns={'property_scalar': 'target'})

mpds_bulk_modulus.to_csv('mpds_bulk_modulus.csv',index=False)
mpds_bulk_modulus.to_csv('mpds_shear_modulus.csv',index=False)

# %% AFLOW DATA

aflow_database = pd.read_csv('master_icsd.csv')

# aflow bulk modulus
aflow_bulk_modulus = aflow_database[['formula','ael_bulk_modulus_vrh']].dropna()
aflow_bulk_modulus = aflow_bulk_modulus.drop_duplicates('formula',keep='last').reset_index(drop=True)
aflow_bulk_modulus = aflow_bulk_modulus.rename(columns={'ael_bulk_modulus_vrh' : 'target'})

# aflow_shear_modulus
aflow_shear_modulus = aflow_database[['formula','ael_shear_modulus_vrh']].dropna()
aflow_shear_modulus = aflow_shear_modulus.drop_duplicates('formula',keep='last').reset_index(drop=True)
aflow_shear_modulus = aflow_shear_modulus.rename(columns={'ael_shear_modulus_vrh' : 'target'})

aflow_bulk_modulus.to_csv('aflow_bulk_modulus.csv',index=False)
aflow_shear_modulus.to_csv('aflow_shear_modulus.csv',index=False)

# %% SEEBECK






