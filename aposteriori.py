import numpy as np
import pandas as pd
from multiprocessing import Pool,cpu_count
from tqdm import tqdm
from atriegc import Trie
from itertools import combinations
from scipy.special import binom

def applyParallel(dfGrouped, func):
    with Pool(cpu_count()) as p:
        ret_list = list(tqdm(p.imap(func, dfGrouped),total=len(dfGrouped)))
    return pd.concat(ret_list)

def arrayParallel(array, func):
    with Pool(cpu_count()) as p:
        ret_list = list(tqdm(p.imap(func, array),total=len(array)))
    return pd.concat(ret_list,ignore_index=True)

def pairwise_evaluation(df,result):
    P_N = binom(df.groupby(['v_gene','j_gene','cdr3_length']).size(),2).sum()
    P = 0
    TP = 0
    for _,family in tqdm(df.groupby(['family']),disable=True):
        for r1,r2 in combinations(family[result],2):
            P += 1
            if r1==r2: TP += 1
    TP_FP = 0
    for _,result in tqdm(df.groupby([result]),disable=True):
        for f1,f2 in combinations(result['family'],2):
            TP_FP += 1
    N = P_N - P
    FP = TP_FP - TP
    if TP_FP==0:
        return 0., 1., 1.
    elif P==0:
        return None, None, None
    else:
        return TP/P, (N-FP)/N, TP/TP_FP  # sensitivity, specificity, precision

def entropy(dfGrouped):
    fs = dfGrouped.size()
    fs = fs/sum(fs)
    return sum(fs*np.log2(fs))

def variation_of_info(df,result='result'):
    VI = entropy(df.groupby(['family'])) + entropy(df.groupby([result])) - 2*entropy(df.groupby([result,'family']))
    return VI

def edit_distance(df,result='family'):
    N = len(df)
    dist = 2*len(df.groupby([result,'family']))-len(df.groupby([result]))-len(df.groupby(['family']))
    return dist/(2*(N-np.sqrt(N)))


def evaluation(args): 
    l,filename1,filename2 = args
    sens_ = []
    spec_ = []
    prec_ = []
    vi_ = []
    dist_ = [] 
    for dataset in np.arange(1,10+1,1):
        df = pd.read_csv(filename1,usecols=['family','v_gene','j_gene','cdr3_length'])
        result = pd.read_csv(filename2, index_col='sequence_id')
        test = pd.concat([df, result], axis=1)
        test['result'] = test['clone']
        test.dropna(inplace=True)
        
        sens,spec,prec = pairwise_evaluation(test)
        vi = variation_of_info(test)
        dist = edit_distance(test)
        
        sens_.append(sens)
        spec_.append(spec)
        prec_.append(prec)
        vi_.append(vi)
        dist_.append(dist)
        
    df = pd.DataFrame(np.array([sens_,spec_,prec_,vi_,dist_]).T,
                      columns=['sensitivity','specificity','precision','var_of_info','edit_distance']) 
    df['cdr3_length'] = l
    return df
