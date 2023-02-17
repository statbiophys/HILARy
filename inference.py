import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster, linkage
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
#from atriegc import TrieNucl as Trie
from atriegc import Trie
from numpy import random

class CDR3Clustering():
    """ Infer families using CDR3 """
    def __init__(self,thresholds,threads=None):
        if threads is None: self.threads = cpu_count()
        else: self.threads = threads
        self.thresholds = thresholds
    
    def applyParallel(self,dfGrouped,func):
        with Pool(self.threads) as p:
            ret_list = list(tqdm(p.imap(func, dfGrouped),total=len(dfGrouped)))
        return pd.concat(ret_list)
    
    def cluster(self,args):
        (v,j,l),df = args
        trie = Trie()
        for cdr3 in df['cdr3']:
            trie.insert(cdr3)
        t = self.thresholds.loc[(self.thresholds.v_gene==v) &
                                (self.thresholds.j_gene==j) &
                                (self.thresholds.cdr3_length==l)].values[0][-1]
        if t>=0:
            dct = trie.clusters(t)
            return df['cdr3'].map(dct)
        else: 
            return df.index.to_frame()
    
    def infer(self,df,group=['v_gene','j_gene','cdr3_length']):
        use = group + ['cdr3']
        df['cluster'] = self.applyParallel(df[use].groupby(group),self.cluster)
        group = group + ['cluster']
        return df.groupby(group).ngroup()+1

class DistanceMatrix():
    def __init__(self,v,j,l,sens,df,threads=None):
        if threads is None: self.threads = cpu_count()
        else: self.threads = threads
        self.l = l
        self.L = 250
        self.l_L = l/self.L
        self.l_L_L = l/(l+self.L)
        
        self.data = df.values
        self.n = self.data.shape[0]
        self.k_max = self.n * (self.n - 1) // 2  # maximum elements in 1D dist array
        self.k_step = self.n ** 2 // 2 // (500)    # ~500 bulks

    def metric(self,arg1,arg2):
        cdr31,s1,n1,i1 = arg1
        cdr32,s2,n2,i2 = arg2
        if i1==i2: return -100.
        else:
            n1n2 = n1*n2
            if n1n2==0: return 100.
            else:
                n = hamming(cdr31,cdr32)
                nL = hamming(s1,s2)
                n0 = (n1+n2 - nL)/2

                exp_n = self.l_L * (nL+1)
                std_n = np.sqrt(exp_n * self.l_L_L)

                exp_n0 = n1n2/self.L
                std_n0 = np.sqrt(exp_n0)

                x = (n - exp_n)/std_n
                y = (n0 - exp_n0)/std_n0
                return x-y  

    def proc(self,start):
        dist = []
        k1 = start
        k2 = min(start + self.k_step, self.k_max)
        for k in range(k1, k2):
            # get (i, j) for 2D distance matrix knowing (k) for 1D distance matrix
            i = int(self.n - 2 - int(np.sqrt(-8 * k + 4 * self.n * (self.n - 1) - 7) / 2.0 - 0.5))
            j = int(k + i + 1 - self.n * (self.n - 1) / 2 + (self.n - i) * ((self.n - i) - 1) / 2)
            # store distance
            a = self.data[i, :]
            b = self.data[j, :]
            d = self.metric(a, b)
            dist.append(d)
        return k1, k2, dist

    def compute(self):
        dist = np.zeros(self.k_max) 
        with Pool(self.threads) as pool:
            for k1, k2, res in tqdm(pool.imap_unordered(self.proc, range(0, self.k_max, self.k_step)),total=500):
                dist[k1:k2] = res
        return dist+100.

class Null():
    """ Infer null x'-y=z distribution and threshold """
    def __init__(self,mutations,lengths=None,default_length=None,
                 model=326713,alignment_length=250,size=1e7):
        if lengths is None: self.lengths = np.arange(15,81+3,3).astype(int)
        else: self.lengths = lengths
        if default_length is None: self.default_length = 45
        else: self.default_length = default_length
        self.model = model
        self.alignment_length = alignment_length
        self.size = int(size)

        bins = np.arange(50+1)
        pni,nis = np.histogram(mutations,bins=bins)
        p = pni[1:]/sum(pni[1:])
        n1s = random.choice(nis[1:-1], size=self.size, replace=True, p=p)
        n2s = random.choice(nis[1:-1], size=self.size, replace=True, p=p)
        exp_n0 = n1s*n2s/self.alignment_length
        n0s = random.poisson(lam=exp_n0, size=self.size)        
        nLs = n1s+n2s-2*n0s
        std_n0 = np.sqrt(exp_n0)
        ys = (n0s - exp_n0)/std_n0
        pns = np.array([self.readNull(l) for l in self.lengths])
        ns = np.array([random.choice(np.arange(l+1),
                                     size=self.size, 
                                     replace=True,
                                     p=pn/pn.sum()) for l,pn in zip(self.lengths,pns)])
        exp_n = np.outer(self.lengths/self.alignment_length,nLs+1)
        std_n = np.sqrt(exp_n * (self.lengths+self.alignment_length)/self.alignment_length)
        xs = (ns - exp_n)/std_n
        zs = xs - ys
        z0s = ns - n0s
        self.z_cdf = np.sort(zs,axis=1)[:int(self.size*0.01)]
        self.z0_cdf = np.sort(z0s,axis=1)[:int(self.size*0.01)]
            
    def readNull(self,l):
        """ Read estimated null distribution """
        cdfs = pd.read_csv('cdfs_{}.csv'.format(self.model))
        cdf = cdfs.loc[cdfs['l']==l].values[0,1:l+1]
        return np.diff(cdf,prepend=[0],append=[1])

    def pRequired(rho,pi=0.99):
        return rho/(1-rho) * (1-pi)/pi
    
    def get_threshold(self,l,rho):
        return self.z_cdf[int(self.size*self.pRequired(rho))]
    
class HILARy():
    """ Infer families using CDR3 and mutations """
    def __init__(self,apriori,threads=None):
        if threads is None: self.threads = cpu_count()
        else: self.threads = threads
        self.group = ['v_gene','j_gene','cdr3_length']
        
        loc = (apriori.classes['v_gene']!="None")
        loc = loc&(apriori.classes['v_gene']!="None")
        loc = loc&(apriori.classes['precise_threshold']<apriori.classes['sensitive_threshold'])
        loc = loc&(apriori.classes['pair_count']>0)
        self.left = apriori.classes.loc[loc].groupby(by).first().pair_count
        
        self.use = ['cdr3','alt_sequence_alignment','nb_mutations','index']
        self.alignment_length=250
        self.x0=0
        self.y0=0
        self.productive = apriori.productive
        
    def applyParallel(self,dfGrouped,func):
        with Pool(self.threads) as p:
            ret_list = list(tqdm(p.imap(func, dfGrouped),total=len(dfGrouped)))
        return pd.concat(ret_list)

    def singleLinkage(self,indices,dist,threshold):
        clusters = fcluster(linkage(dist,
                                    method='single',
                                    metric='precomputed'),
                            criterion='distance',
                            t=threshold)
        return {i:c for i,c in zip(indices, clusters)}

    def class2pairs(self.args):
        (v,j,l,_),df = args
        indices = np.unique(df['precise_cluster'])
        if len(indices)>1:
            translateIndices = dict(zip(indices,range(len(indices))))
            df['index'] = df['precise_cluster'].map(translateIndices)
            dim = len(indices)
            distanceMatrix = np.ones((dim,dim),dtype=float)*(200)
            for i in range(dim): distanceMatrix[i,i]=0

            for (cdr31,s1,n1,i1),(cdr32,s2,n2,i2) in combinations(df[use].values,2):
                if i1!=i2:
                    n1n2 = n1*n2
                    if n1n2>0:
                        n = hamming(cdr31,cdr32)
                        nL = hamming(s1,s2)
                        n0 = (n1+n2 - nL)/2

                        exp_n = l/alignment_length * (nL+1)
                        std_n = np.sqrt(exp_n * (l+alignment_length)/alignment_length)

                        exp_n0 = n1n2/alignment_length
                        std_n0 = np.sqrt(exp_n0)

                        x = (n - exp_n)/std_n
                        y = (n0 - exp_n0)/std_n0
                        distance = x-y+100

                        distanceMatrix[i1,i2] = distance
                        distanceMatrix[i2,i1] = distance

            sl = singleLinkage(indices,distanceMatrix,threshold=x0-y0+100)
            return df['precise_cluster'].map(sl)
        else:
            return df['precise_cluster']
    
    def mark_class(self,args):
        _,df = args
        return df['to_resolve']

    def to_do(self,df,size_threshold=1e3):
        df['to_resolve'] = True
        df['to_resolve'] = self.applyParallel([(g,df.groupby(self.group).get_group(g)) for g in left.index],
                                              self.mark_class)
        df.fillna(value={'to_resolve': False}, inplace=True)
        dfGrouped = df.loc[df['to_resolve']].groupby(self.group + ['sensitive_cluster'])
        sizes = dfGrouped.size()
        mask = sizes>size_threshold
        self.large_to_do = sizes[mask].index
        self.small_to_do = sizes[~mask].index

        
    def infer(self,df):
        dfGrouped = df.loc[self.productive].groupby(['v_gene',
                                           'j_gene',
                                           'cdr3_length',
                                           'sensitive_cluster'])
        df['family_cluster'] = applyParallel([(g,dfGrouped.get_group(g)) for g in self.small_to_do],class2pairs)
        for g in large:
            dm = DistanceMatrix(*g,dfGrouped.get_group(g)[['cdr3','alt_sequence_alignment','nb_mutations','precise_cluster']])
            d = dm.compute()
            shift = min(d)
            dct = self.singleLinkage(dfGrouped.get_group(g).index,d+shift,0.1+shift)
            df['family_cluster'] = df.index.map(dct)