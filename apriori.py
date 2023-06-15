import os
import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)
from scipy.special import binom,erf
from itertools import combinations
from textdistance import hamming
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from scipy.special import factorial
from scipy import interpolate

def applyParallel(dfGrouped, func, silent=False, cpuCount=None):
    if cpuCount is None: cpuCount = cpu_count()
    with Pool(cpuCount) as p:
        ret_list = list(tqdm(p.imap(func, dfGrouped),total=len(dfGrouped),disable=silent))
    return pd.concat(ret_list)

def count_mutations(args):
    _,df = args
    return df[['alt_sequence_alignment','alt_germline_alignment']].apply(lambda x: hamming(*x), axis=1)

def preprocess(df,max_mutation_count=60):
    usecols = ['sequence_id',
               'v_gene',
               'j_gene',
               'cdr3_length',
               'cdr3',
               'alt_sequence_alignment',
               'alt_germline_alignment',
               'mutation_count']

    df[['v_gene','_']] = df['v_call'].str.split('*',1,expand=True) 
    df[['j_gene','_']] = df['j_call'].str.split('*',1,expand=True) 
    df['cdr3'] = df['junction'].str[3:-3]
    df['cdr3_length'] = df['cdr3'].str.len()
    df = df.loc[df['cdr3_length']>0]
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['alt_sequence_alignment'] = df['v_sequence_alignment'] + df['j_sequence_alignment']
    df['alt_germline_alignment'] = df['v_germline_alignment'] + df['j_germline_alignment']
    df['mutation_count'] = applyParallel(df.groupby(['v_gene',
                                                     'j_gene',
                                                     'cdr3_length']), count_mutations)

    return df.loc[df['mutation_count']<=max_mutation_count][usecols].astype({'mutation_count': int,
                                                                             'cdr3_length': int})

class EM():
    """ Finds the mixture distribution
        that explains the statistics of distances """
    def __init__(self,l,h,model=326713,howMany=10,positives='geometric'):
        self.l = int(l)
        self.h = h.astype(int)[:self.l+1]
        self.b = np.arange(self.l+1,dtype=int)
        self.model = model
        self.constP0 = self.readNull()
        self.howMany = howMany
        self.positives = positives
            
    def readNull(self):
        """ Read estimated null distribution """
        dirname = os.path.dirname(__file__)
        cdfs = pd.read_csv(dirname + '/data/cdfs_{}.csv'.format(self.model))
        cdf = cdfs.loc[cdfs['l']==self.l].values[0,1:self.l+1]
        return np.diff(cdf,prepend=[0],append=[1])
    
    def discreteExpectation(self,theta):
        """ Calculate membership probabilities """
        rho,mu = theta
        if self.positives == 'geometric': P1 = rho/(mu+1) * (mu/(mu+1))**self.b
        elif self.positives == 'poisson': P1 = rho * mu**self.b * np.exp(-mu) / factorial(self.b)
        P0 = (1-rho) * self.constP0[self.b]
        return np.array([P1,P0])/(P1+P0)

    def dicreteMaximization(self,theta):
        """ Maximize current likelihood """
        P1, P0 = self.discreteExpectation(theta)
        P1Sum, P0Sum = (self.h*P1).sum(), (self.h*P0).sum()
        rho = P1Sum/(P1Sum+P0Sum)
        mu = np.dot(self.h*P1,self.b)/P1Sum
        return rho,mu

    def discreteEM(self):
        """ EM with discrete model:
            P1 geometric or Poisson with mean mu
            P0 estimated with Ppost """
        mu = 0.02*self.l
        rho = self.h[0]/sum(self.h)*(1+mu)
        theta = (rho,mu)
        for i in range(self.howMany): theta = self.dicreteMaximization(theta)
        return theta
    
    def discreteMix(self,x,theta):
        """ Evaluate mixture distribution """
        rho,mu = theta
        if self.positives == 'geometric': 
            return rho/(mu+1) * (mu/(mu+1))**x + (1-rho)*self.constP0[x]
        if self.positives == 'poisson':   
            return rho * mu**x * np.exp(-mu) / factorial(x) + (1-rho)*self.constP0[x]
    
    def error(self,theta,threshold=0.15,error='rrmse'):
        """ Estimate goodness of fit
            By default returns rescaled root MSE """
        y1 = self.h/sum(self.h)
        y2 = self.discreteMix(self.b,theta)
        mask = (y1>0) * (self.b<threshold*self.l)
        y1m, y2m = y1[mask], y2[mask]
        logy1, logy2 = np.log(y1m), np.log(y2m)
        
        mse = ((y1m-y2m)**2).sum()/mask.sum()
        rmse = np.sqrt(mse)
        msle = ((logy1-logy2)**2).sum()/mask.sum()
        rmsle = np.sqrt(msle)
        mae = np.abs(y1m-y2m).sum()/mask.sum()
        dkl = (y2m * (logy2-logy1)).sum()/np.log(2)
        if error=='rmse': return rmse
        elif error=='rmsle': return rmsle
        elif error=='mae': return mae
        elif error=='dkl': return dkl
        elif error=='rrmse': return rmse/theta[0]
        else: return False


class Apriori():
    """ Computes statistics of pairwise distances """
    
    def __init__(self, df, lengths=None, default_length=None, nmin=1e5, nmax=1e5, threads=None):
        if lengths is None: self.lengths = np.arange(15,81+3,3).astype(int)
        else: self.lengths = lengths
        if default_length is None: self.default_length = 45
        else: self.default_length = default_length
        self.bins = range(self.lengths[-1]+2)
        self.nmin = int(nmin)
        self.nmax = int(nmax)
        if threads is None: self.threads = cpu_count()-1
        else: self.threads = threads

        self.productive = (df['cdr3_length']%3==0)&(df['cdr3_length']>=15)
        self.loc = df.loc[self.productive]['cdr3_length'].isin(self.lengths)
        self.by = ['v_gene','j_gene','cdr3_length']
        self.use = self.by + ['cdr3']
        
        classes = df.loc[self.productive].groupby(self.by)
        vjl_sequence_count = classes.size()
        vjl_pair_count = vjl_sequence_count.apply(self.nbOfPairs).astype(int)
        l_sequence_count = vjl_sequence_count.groupby('cdr3_length').sum()
        l_pair_count = vjl_pair_count.groupby('cdr3_length').sum()
        
        indices = np.array(vjl_sequence_count.index.tolist())
        vs = indices[:,0].astype(str)
        js = indices[:,1].astype(str)
        ls = indices[:,2].astype(int)
        extra_ls = np.array(l_sequence_count.index.tolist())
        extra_vs = np.array([None]*len(extra_ls))
        extra_js = extra_vs
        
        columns = self.by + ['sequence_count','pair_count']
        VJlclasses = pd.DataFrame(np.array([vs,js,ls,
                                            vjl_sequence_count,
                                            vjl_pair_count]).T,columns=columns)
        lclasses = pd.DataFrame(np.array([extra_vs,extra_js,extra_ls,
                                          l_sequence_count,
                                          l_pair_count]).T,columns=columns)
        self.classes = pd.concat([VJlclasses,lclasses],
                                 ignore_index=True).astype({'v_gene': 'str',
                                                            'j_gene': 'str',
                                                            'cdr3_length': 'int',
                                                            'sequence_count': 'int',
                                                            'pair_count': 'int'})
        self.classes.sort_values('sequence_count',ascending=False,inplace=True)
        self.classes['class_id'] = range(1,len(self.classes)+1)
        self.classes.reset_index(drop=True, inplace=True)

    def nbOfPairs(self,x):
        return binom(x,2)
    
    def vjl2x(self,args):
        """ Compute CDR3 hamming distances within VJl class """
        (v,j,l),df = args
        xs = []
        for s1,s2 in combinations(df['cdr3'].values,2):
            xs.append(hamming(s1,s2))
        x = pd.DataFrame()
        x['x'] = xs
        return x

    def l2x(self,df,l):
        """ Compute histogram of distances within l class """
        x = applyParallel(df.groupby(self.by),
                          self.vjl2x,
                          silent=True,
                          cpuCount=self.threads)
        return np.histogram(x.x,bins=self.bins,density=False)[0]

    def computeAlll(self,df):
        """ Compute histograms for all large l classes """  
        histograms = []
        class_ids = []
        loc = self.classes['cdr3_length'].isin(self.lengths) 
        loc = loc& (self.classes['v_gene']=='None') 
        loc = loc& (self.classes['pair_count']>=self.nmin)
        for _,row in tqdm(self.classes.loc[loc].iterrows()):
            frac = min(np.sqrt(self.nmax/row.pair_count),1)
            h = self.l2x(df[self.use].loc[df['cdr3_length']==row.cdr3_length].sample(frac=frac),row.cdr3_length)
            histograms.append(h)
            class_ids.append(row.class_id)
        results = pd.DataFrame(np.array(histograms),index=class_ids)
        results['class_id'] = results.index
        return results
        
    def vjls2x(self,args):
        """ Compute histogram of distances within VJl class """
        i,df = args
        xs = []
        for s1,s2 in combinations(df['cdr3'].values,2):
            xs.append(hamming(s1,s2))
        return pd.DataFrame(np.histogram(xs,bins=self.bins,density=False)[0],columns=[i]).transpose()
    
    def computeAllVJl(self,df):
        """ Compute histograms for all large VJl classes """
        loc = self.classes['cdr3_length'].isin(self.lengths) 
        loc = loc& (self.classes['v_gene']!='None') 
        loc = loc& (self.classes['pair_count']>=self.nmin)
        groups = df.loc[self.loc].groupby(self.by)
        results = applyParallel([(row.class_id,
                                  groups.get_group((row.v_gene,
                                                    row.j_gene,
                                                    row.cdr3_length)).sample(frac=min(np.sqrt(self.nmax/row.pair_count),
                                                                                      1))) for _,row in self.classes.loc[loc].iterrows()],
                                self.vjls2x,
                                cpuCount=self.threads)
        results['class_id'] = results.index
        return results
    
    def get_histograms(self,df):
        """ Compute histograms for all large classes """
        hs_l = self.computeAlll(df)
        hs_vjl = self.computeAllVJl(df)
        self.histograms = pd.concat([hs_l,hs_vjl],ignore_index=True).sort_values('class_id')[['class_id']+list(range(81+1))]
            
    def estimate(self,args):
        class_id,h = args
        l = self.classes.loc[self.classes.class_id==class_id].cdr3_length
        em = EM(int(l),h.values[0,1:],positives='geometric')    
        rho_geo,mu_geo = em.discreteEM()
        error_geo = em.error([rho_geo,mu_geo])
        em = EM(int(l),h.values[0,1:],positives='poisson')    
        rho_poisson,mu_poisson = em.discreteEM()
        error_poisson = em.error([rho_poisson,mu_poisson])
        result = pd.DataFrame(columns=['class_id',
                                       'rho_geo',
                                       'mu_geo',
                                       'error_geo',
                                       'rho_poisson',
                                       'mu_poisson',
                                       'error_poisson'])
        result.class_id = [class_id]
        result.rho_geo = [rho_geo]
        result.mu_geo = [mu_geo]
        result.error_geo = [error_geo]
        result.rho_poisson = [rho_poisson]
        result.mu_poisson = [mu_poisson]
        result.error_poisson = [error_poisson]
        return result

    def assign_prevalence(self,args):
        l,ldf = args
        p = ldf.loc[ldf['v_gene']=='None'].prevalence.values[0]
        if np.isnan(p): p = self.mean_prevalence
        return ldf[['prevalence']].fillna(p)
 
    def assign_mean_distance(self,args):
        l,ldf = args
        m = ldf.loc[ldf['v_gene']=='None'].mean_distance.values[0]
        if np.isnan(m): m = self.mean_mean_distance
        return ldf[['mean_distance']].fillna(m)

    def get_parameters(self):
        self.parameters = applyParallel(self.histograms.groupby(['class_id']),
                                        self.estimate,
                                        cpuCount=self.threads).reset_index(drop=True)
        self.classes.index=self.classes.class_id
        self.parameters.index=self.parameters.class_id
        self.classes['prevalence'] = self.parameters[['rho_poisson','rho_geo']].min(axis=1)
        self.classes['mean_distance'] = self.parameters['mu_poisson']/self.classes['cdr3_length']#^
        
        loc = (self.classes['v_gene']=='None') 
        loc = loc& (~self.classes['prevalence'].isna()) 
        loc = loc& (~self.classes['mean_distance'].isna())
        ns = self.classes.loc[loc]['pair_count']
        rhos = self.classes.loc[loc]['prevalence']
        mus = self.classes.loc[loc]['mean_distance']
        ws = ns*(ns-1)
        ws = ws/sum(ws)
        self.mean_prevalence = sum(ws*rhos)
        self.mean_mean_distance = sum(ws*mus)
        
        self.classes['effective_prevalence'] = applyParallel(self.classes.groupby(['cdr3_length']),
                                                             self.assign_prevalence,
                                                             cpuCount=self.threads)
        self.classes['effective_mean_distance'] = applyParallel(self.classes.groupby(['cdr3_length']),
                                                                self.assign_mean_distance,
                                                                cpuCount=self.threads)
        
    def assign_precise_thresholds(self,args):
        l,ldf = args
        rhos = ldf['effective_prevalence']
        mus = ldf['effective_mean_distance']*l
        bins = np.arange(l+1)
        if l in self.lengths:    
            cdf0 = self.cdfs.loc[self.cdfs['l']==l].values[0,1:l+2] 
        else:
            default_cdf0 = self.cdfs.loc[self.cdfs['l']==self.default_length].values[0,1:self.default_length+2] 
            default_bins = np.arange(self.default_length+1)/self.default_length
            f = interpolate.interp1d(default_bins, default_cdf0, fill_value='extrapolate')
            cdf0 = f(bins/l)
        cdf1 = (np.array([mu**bins* np.exp(-mu) for mu in mus])/factorial(bins)).cumsum(axis=1)
        ps = cdf0/cdf1
        t_prec = np.array([p<rho/(1-rho)*(1-self.precision)/self.precision for p,rho in zip(ps,rhos)]).sum(axis=1)-1
        t_sens = (cdf1<self.sensitivity).sum(axis=1)
        ldf['precise_threshold'] = np.min([t_prec,t_sens],axis=0)
        return ldf['precise_threshold'] 

    def assign_sensitive_thresholds(self,args):
        l,ldf = args
        mus = ldf['effective_mean_distance']*l
        bins = np.arange(l+1)
        cdf1 = (np.array([mu**bins* np.exp(-mu) for mu in mus])/factorial(bins)).cumsum(axis=1)
        t_sens = (cdf1<self.sensitivity).sum(axis=1)
        ldf['sensitive_threshold'] = t_sens
        return ldf['sensitive_threshold'] 

    def get_thresholds(self,precision=0.99,sensitivity=0.9,model=326713):
        self.precision = precision
        self.sensitivity = sensitivity
        
        dirname = os.path.dirname(__file__)
        self.cdfs = pd.read_csv(dirname + '/data/cdfs_{}.csv'.format(model))
        
        self.classes['precise_threshold'] = applyParallel(self.classes.groupby(['cdr3_length']),
                                                          self.assign_precise_thresholds,
                                                          cpuCount=self.threads)
        self.classes['sensitive_threshold'] = applyParallel(self.classes.groupby(['cdr3_length']),
                                                            self.assign_sensitive_thresholds,
                                                            cpuCount=self.threads)