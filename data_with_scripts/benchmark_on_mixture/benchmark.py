import os

import numpy as np
import pandas as pd

from hilary.apriori import Apriori
from hilary.inference import HILARy
from hilary.utils import create_classes, pairwise_evaluation


# here some code to infer clonal families.
def infer_hilary(dataframe_processed,model='human_B_heavy',fast=False,precision=0.99,sensitivity=0.95,null_model='vjl'):
    df=dataframe_processed.copy()
    apriori = Apriori(silent=False, threads=-1, precision=precision, sensitivity=sensitivity,model=model,null_model=null_model)
    apriori.classes = create_classes(df)
    apriori.get_histograms(df)
    apriori.get_parameters()
    hilary = HILARy(apriori, df=df)
    df1=hilary.compute_crude_method_clusters(df,normalized_threshold=0.1)
    dataframe_cdr3 = hilary.compute_prec_sens_clusters(df=df1)
    if fast: return dataframe_cdr3,apriori,0
    df_out=dataframe_cdr3.copy()
    hilary.get_xy_thresholds(df=dataframe_cdr3)
    hilary.classes["xy_threshold"] = hilary.classes["xy_threshold"]
    df_full=hilary.infer(df=dataframe_cdr3,size_threshold=100000)
    return df_out,apriori,df_full

def f(i,sens,prec,data):
    if i=='vjl': return infer_hilary(data,null_model='vjl',sensitivity=sens,precision=prec)
    if i=='jl': return infer_hilary(data,null_model='jl',sensitivity=sens,precision=prec)
    if i=='l': return infer_hilary(data,null_model='l',sensitivity=sens,precision=prec)
    if i=='crude': return infer_hilary(data,sensitivity=sens,precision=prec,fast=True)
print('load_data')
productive=pd.read_csv('simulated_cfs_mixture.csv.gz')
productive['sequence_id']=range(len(productive))
productive=productive.sample(frac=1)
sens=0.995
prec=0.999
ns=[int(5e4),int(1e5),int(2e5),int(5e5),int(1e6)]

for n in ns:
    if not os.path.exists(f'evaluation_{int(n)}.csv.gz'):
        out=pd.DataFrame()
        print('size:',n)
        #mutated=productive.loc[productive.mutation_count>0]
        n_reps= 3
        for k in range(n_reps):
            print('rep:',k)
            data=productive.sample(n)
            for model in ['l','vjl','jl']:
                print(model)
                data_seqs_CFs,apriori_data,data_seqs_CFs_full=f(model,sens,prec,data)
                a,b=pairwise_evaluation(data_seqs_CFs_full, truth='new_family', partition='clone_id')
                e,f_=pairwise_evaluation(data_seqs_CFs_full, truth='new_family', partition = 'crude_method_family')
                g,h=pairwise_evaluation(data_seqs_CFs_full, truth='new_family', partition='precise_cluster')
                results=pd.DataFrame({  'n':[n],
                                        'precision_clone_id':[a],
                                        'recall_clone_id':[b],
                                        'precision_crude_method_family':[e],
                                        'recall_crude_method_family':[f_],
                                        'precision_precise_cluster':[g],
                                        'recall_precise_cluster':[h],'method':[model]})
                out=pd.concat([out,results])
        out.to_csv(f'evaluation_{int(n)}.csv.gz',index=False,compression='gzip')
    else:
        print(n,'file already exists')
