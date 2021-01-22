import numpy as np
import pandas as pd

def calculate_data_shape(x):
    return x.shape
    
def take_columns(x):
    return x.columns
 
def calculate_target_ratio(x, target_name):
    return round(x[target_name].mean(), 2)

def calculate_data_dtypes(x):
    dt = x.dtypes.value_counts()
    return [dt[0]+dt[1], dt[2]]
 
def calculate_cheap_apartment(x):
    return len(x[x['price_doc'] <= 1000000])
   
def calculate_squad_in_cheap_apartment(x):
    return round(x[x['price_doc'] <= 1000000]['full_sq'].mean(),0)
    
def calculate_mean_price_in_new_housing(x):
    return round(x.loc[(x['build_year'] >= 2010) & (x['num_room'] == 3)]['price_doc'].mean(),0)
    
def calculate_mean_squared_by_num_rooms(x):
    return np.around(x.groupby(['num_room'])['full_sq'].mean(), decimals = 2)
   
def calculate_squared_stats_by_material(x):
    data1 = np.around(x.groupby(['material'])['full_sq'].max(), decimals = 2)
    data2 = np.around(x.groupby(['material'])['full_sq'].min(), decimals = 2)
    res = pd.concat([data1, data2], axis = 1)
    res.columns = ['amax', 'amin']
    return res
    
def calculate_crosstab(x):
    return np.around(x.pivot_table('price_doc',  index = 'sub_area', columns = 'product_type', fill_value = 0, aggfunc = np.mean), 2)
