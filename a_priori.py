import numpy as np
import pandas as pd
from scipy.special import binom,erf
from itertools import combinations
from textdistance import hamming
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from scipy.special import factorial

def applyParallel(dfGrouped, func, silent=False):
    with Pool(cpu_count()) as p:
        ret_list = list(tqdm(p.imap(func, dfGrouped),total=len(dfGrouped),disable=silent))
    return pd.concat(ret_list)

class Histograms():
    """ Computes statistics of pairwise distances """
    
    def __init__(self, df, lengths=None, nmin=1e5, nmax=1e5):
        if lengths is None: self.lengths = np.arange(15,81+3,3).astype(int)
        else: self.lengths = lengths
        self.bins = range(self.lengths[-1]+2)
        self.nmin = int(nmin)
        self.nmax = int(nmax)
        
        self.loc = df['cdr3_length'].isin(self.lengths)
        self.by = ['v_gene','j_gene','cdr3_length']
        self.use = self.by + ['cdr3']
        classes = df.loc[self.loc].groupby(self.by)
        self.VJl_sizes = classes.size().sort_values(ascending=False).apply(self.nbOfPairs)
        self.l_sizes = self.VJl_sizes.groupby('cdr3_length').sum().sort_values(ascending=False)
        
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
                          silent=True)
        return np.histogram(x.x,bins=self.bins,density=False)[0]

    def computeAlll(self,df):
        """ Compute histograms for all large l classes """
        hs = []
        ls = []
        large = self.l_sizes>self.nmin
        for l,size in tqdm(self.l_sizes[large].to_frame().iterrows(),total=sum(large)):
            frac = min(np.sqrt(self.nmax/size[0]),1)
            h = self.l2x(df[self.use].loc[df['cdr3_length']==l].sample(frac=frac),l)
            hs.append(h)
            ls.append(l)
        res = pd.DataFrame(hs)
        res['cdr3_length'] = ls 
        res['v_gene'] = None 
        res['j_gene'] = None 
        return res
    
    def vjls2x(self,args):
        """ Compute histogram of distances within VJl class """
        i,df = args
        xs = []
        for s1,s2 in combinations(df['cdr3'].values,2):
            xs.append(hamming(s1,s2))
        return pd.DataFrame(np.histogram(xs,bins=self.bins,density=False)[0],columns=[i]).transpose()
    
    def computeAllVJl(self,df):
        """ Compute histograms for all large VJl classes """
        target = self.VJl_sizes[self.VJl_sizes>self.nmin].to_frame()
        classes = df.loc[self.loc].groupby(self.by)
        res = applyParallel([(i,classes.get_group(vjl).sample(frac=min(np.sqrt(self.nmax/size[0]),1))) for i,(vjl,size) in enumerate(target.iterrows())], self.vjls2x)
        target.reset_index(inplace=True)
        res['cdr3_length'] = target['cdr3_length'] 
        res['v_gene'] = target['v_gene']
        res['j_gene'] = target['j_gene']
        return res
    
    def estimate(self,df):
        """ Compute histograms for all large classes """
        hs_l = self.computeAlll(df)
        hs_vjl = self.computeAllVJl(df)
        self.hs = pd.concat([hs_l,hs_vjl],ignore_index=True)
        
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
        cdfs = pd.read_csv('cdfs_{}.csv'.format(self.model))
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
        if self.positives == 'geometric': return rho/(mu+1) * (mu/(mu+1))**x + (1-rho)*self.constP0[x]
        if self.positives == 'poisson':   return rho * mu**x * np.exp(-mu) / factorial(x) + (1-rho)*self.constP0[x]
    
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