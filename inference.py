import pandas as pd
import numpy as np
from multiprocessing import Pool,cpu_count
from tqdm import tqdm

from textdistance import hamming
from itertools import combinations
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster, linkage

from EM import EM
from atriegc import Trie

def pRequired(rho,l,mu=None,pi=0.99):
    if mu is None: mu = 0.05*l 
    return rho/(1-rho) * (1-pi)/pi * (1-(mu/(mu+1))**(np.arange(l+1)+1))

class CDR3Clustering():
    """ Infer clonal families with threshold """
    def __init__(self,threshold,threads=None):
        if threads is None: self.threads = cpu_count()
        else: self.threads = threads
        self.threshold = threshold
    
    def applyParallel(self,dfGrouped, func):
        with Pool(self.threads) as p:
            ret_list = list(tqdm(p.imap(func, dfGrouped),total=len(dfGrouped)))
        return pd.concat(ret_list)
    
    def cluster(self,args):
        (v,j,l),df = args
        trie = Trie()
        for cdr3 in df['CDR3']:
            trie.insert(cdr3)
        t = self.threshold
        if t>=0:
            dct = trie.clusters(t)
            return df['CDR3'].map(dct)
        else: 
            return df.index.to_frame()
    
    def infer(self,df,group=['V_GENE','J_GENE','CDR3_LENGTH']):
        #assert df['CDR3_LENGTH'].dtype=='int'
        #df['ID'] = df.index
        use = group + ['CDR3']
        df['CLUSTER'] = self.applyParallel(df[use].groupby(group),self.cluster)
        group = group + ['CLUSTER']
        return df.groupby(group).ngroup()+1
        
    
class Inference():
    """ Infer clonal families """
    def __init__(self,l,precision=0.99,data='B',threads=None,sizeThreshold=5e3,nb_bulks=500):
        self.l = l
        self.pi = precision
        if data=='B': self.start = 3*27
        elif data=='Full': self.start = 0
        if threads is None: self.threads = cpu_count()
        else: self.threads = threads
        self.sizeThreshold = sizeThreshold
        self.nb_bulks = nb_bulks
        
        self.cys104 = 3*104
        self.try118 = self.cys104+self.l
        self.L = 3*115-self.start
        self.Lfrac = self.l/self.L
        self.Lharm = (self.l+self.L)/self.L
        self.nt2nb = {'A':0,'C':1,'G':2,'T':3,'.':4,'-':5}
        self.x0 = -1

    def applyParallel(self,dfGrouped, func):
        with Pool(self.threads) as p:
            ret_list = list(tqdm(p.imap(func, dfGrouped),total=len(dfGrouped)))
        return pd.concat(ret_list)

    def mutations(self,germline,sequence):
        ms = []
        for x,(nt1,nt2) in enumerate(zip(germline,sequence)):
            if nt1!=nt2: ms.append((x,self.nt2nb[nt2]))
        return set(ms)

    def mutationsParallel(self,args):
        _,df = args
        return df.apply(lambda x: self.mutations(x.GERMLINE, x.SEQUENCE), axis=1)

    def preprocess(self,df):
        use = ['V_CALL','J_CALL','JUNCTION','GERMLINE_IMGT','SEQUENCE_IMGT']
        df.dropna(subset=use,inplace=True)

        df['ID']=df.index
        df[['V_GENE','_']] = df['V_CALL'].str.split('*',1,expand=True) 
        df[['J_GENE','_']] = df['J_CALL'].str.split('*',1,expand=True) 
        df['CDR3'] = df['JUNCTION'].str[3:-3]
        df['CDR3_LENGTH'] = df['CDR3'].str.len()
        df['GERMLINE'] = df['GERMLINE_IMGT'].str[self.start:self.cys104] + df['GERMLINE_IMGT'].str[self.try118:]
        df['SEQUENCE'] = df['SEQUENCE_IMGT'].str[self.start:self.cys104] + df['SEQUENCE_IMGT'].str[self.try118:]

        use = ['V_GENE','J_GENE','CDR3','CDR3_LENGTH','SEQUENCE','GERMLINE']
        df.dropna(subset=use,inplace=True)

        #df['MUTATIONS'] = df.apply(lambda x: self.mutations(x.GERMLINE, x.SEQUENCE), axis=1)
        df['MUTATIONS'] = self.applyParallel(df[['GERMLINE','SEQUENCE']].groupby(df.index%self.nb_bulks), self.mutationsParallel)
        df['NB_MUT'] = df.apply(lambda x: len(x.MUTATIONS), axis=1)

        return df.loc[df['NB_MUT']<0.5*self.L]

    def preprocessWithGenes(self,df):
        use = ['V_GENE', 'J_GENE', 'CDR3_LENGTH', 'CDR3', 'SEQUENCE_IMGT', 'GERMLINE_IMGT', 'ID']
        df.dropna(subset=use,inplace=True)

        df['GERMLINE'] = df['GERMLINE_IMGT'].str[self.start:self.cys104] + df['GERMLINE_IMGT'].str[self.try118:]
        df['SEQUENCE'] = df['SEQUENCE_IMGT'].str[self.start:self.cys104] + df['SEQUENCE_IMGT'].str[self.try118:]

        use = ['V_GENE','J_GENE','CDR3','CDR3_LENGTH','SEQUENCE','GERMLINE']
        df.dropna(subset=use,inplace=True)

        #df['MUTATIONS'] = df.apply(lambda x: self.mutations(x.GERMLINE, x.SEQUENCE), axis=1)
        df['MUTATIONS'] = self.applyParallel(df[['GERMLINE','SEQUENCE']].groupby(df.index%self.nb_bulks), self.mutationsParallel)
        df['NB_MUT'] = df.apply(lambda x: len(x.MUTATIONS), axis=1)

        return df.loc[df['NB_MUT']<0.5*self.L]
    
    def preclustering(self,df,t_prec=None,t_sens=None):
        em = EM(df,self.l)
        em.infer()
        self.rho,mu = em.theta
        ps = pRequired(self.rho,self.l,mu=None,pi=self.pi)
        self.y0 = -np.log10(ps[-1])-3
        self.x0_y0 = self.x0 - self.y0
        
        if t_prec is None: 
            self.t_prec = int(sum(np.cumsum(em.constP0)<ps))-1
        else:
            self.t_prec = t_prec
        if t_sens is None: 
            self.t_sens = int(self.l*0.1)+1
        else:
            self.t_sens = t_sens
            
        if self.t_prec>=self.t_sens:
            print('High-sensitivity clustering is enough, {}>={}'.format(self.t_prec,self.t_sens))
            sensClustering = CDR3Clustering(threshold=self.t_sens)
            df['SENS_CLUSTER'] = sensClustering.infer(df)
            return df['SENS_CLUSTER'], df['SENS_CLUSTER']
        else:
            sensClustering = CDR3Clustering(threshold=self.t_sens)
            precClustering = CDR3Clustering(threshold=self.t_prec)
            df['SENS_CLUSTER'] = sensClustering.infer(df)
            df['PREC_CLUSTER'] = precClustering.infer(df)
            return df['SENS_CLUSTER'], df['PREC_CLUSTER']
        
    def singleLinkage(self,indices,distanceMatrix,threshold):
        clusters = fcluster(linkage(squareform(distanceMatrix),
                                    method='single',
                                    metric='precomputed'),
                            criterion='distance',
                            t=threshold)
        return {i:c for i,c in zip(indices, clusters)}

    def singleLinkage1D(self,indices,dist,threshold):
        clusters = fcluster(linkage(dist,
                                    method='single',
                                    metric='precomputed'),
                            criterion='distance',
                            t=threshold)
        return {i:c for i,c in zip(indices, clusters)}    
    
    def class2pairs(self,args):
        _,df = args
        indices = np.unique(df['PREC_CLUSTER'])
        if len(indices)>1:
            translateIndices = dict(zip(indices,range(len(indices))))
            df['INDEX'] = df['PREC_CLUSTER'].map(translateIndices)
            dim = len(indices)
            distanceMatrix = np.ones((dim,dim),dtype=float)*200
            for i in range(dim): distanceMatrix[i,i]=0
            use = ['CDR3','MUTATIONS','NB_MUT','INDEX']
            for (cdr31,m1,n1,i1),(cdr32,m2,n2,i2) in combinations(df[use].values,2):
                if i1!=i2:
                    n1n2 = n1*n2
                    if n1n2>0:
                        n = hamming(cdr31,cdr32)
                        n0 = len(m1.intersection(m2))
                        nL = n1+n2 - 2*n0
                        n0 = (n1+n2 - nL)/2

                        exp_n = self.Lfrac * (nL+1)
                        std_n = np.sqrt(exp_n * self.Lharm)

                        exp_n0 = n1n2/self.L
                        std_n0 = np.sqrt(exp_n0)

                        x = (n - exp_n)/std_n
                        y = (n0 - exp_n0)/std_n0
                        distance = x-y+100

                        distanceMatrix[i1,i2] = distance
                        distanceMatrix[i2,i1] = distance

            sl = self.singleLinkage(indices,distanceMatrix,threshold=self.x0_y0+100)
            return df['PREC_CLUSTER'].map(sl)
        else:
            return df['PREC_CLUSTER']
 
    def metric(self,args):
        (cdr31,m1,n1,i1),(cdr32,m2,n2,i2) = args
        if i1==i2: return 0.
        else:
            n1n2 = n1*n2
            if n1n2==0: return 200.
            else:
                n = hamming(cdr31,cdr32)
                n0 = len(m1.intersection(m2))
                nL = n1+n2 - 2*n0
                n0 = (n1+n2 - nL)/2

                exp_n = self.Lfrac * (nL+1)
                std_n = np.sqrt(exp_n * self.Lharm)

                exp_n0 = n1n2/self.L
                std_n0 = np.sqrt(exp_n0)

                x = (n - exp_n)/std_n
                y = (n0 - exp_n0)/std_n0
                distance = x-y+100
                return distance

    def proc(self,start):
        dist = []
        k1 = start
        k2 = min(start + self.k_step, self.k_max)
        for k in range(k1, k2):
            # get (i, j) for 2D distance matrix knowing (k) for 1D distance matrix
            i = int(self.n - 2 - int(np.sqrt(-8 * k + 4 * self.n * (self.n - 1) - 7) / 2.0 - 0.5))
            j = int(k + i + 1 - self.n * (self.n - 1) / 2 + (self.n - i) * ((self.n - i) - 1) / 2)
            # store distance
            d = self.metric((self.data[i, :], self.data[j, :]))
            dist.append(d)
        return k1, k2, dist

    def XY(self,df):
        by = ['SENS_CLUSTER']
        dfGrouped=df.groupby(by)
        sizes = dfGrouped.size()
        mask = sizes>self.sizeThreshold
        self.large = sizes[mask].index
        self.small = sizes[~mask].index
        
        use = ['CDR3','MUTATIONS','NB_MUT','PREC_CLUSTER']
        df['FAMILY_CLUSTER'] = self.applyParallel([(g,dfGrouped.get_group(g)[use]) for g in self.small],self.class2pairs)
        
        for g in self.large:
            self.data = dfGrouped.get_group(g)[use].values
            self.n = self.data.shape[0]
            self.k_max = self.n * (self.n - 1) // 2  # maximum elements in 1D dist array
            self.k_step = self.n ** 2 // 2 // self.nb_bulks    # ~500 bulks
            dist = np.zeros(self.k_max) 
            with Pool(self.threads) as pool:
                for k1, k2, res in tqdm(pool.imap_unordered(self.proc, range(0, self.k_max, self.k_step)),total=self.nb_bulks):
                    dist[k1:k2] = res
            sl = self.singleLinkage1D(dfGrouped.get_group(g).index,dist,threshold=self.x0_y0+100)
            indices = dfGrouped.get_group(g).index
            df.loc[indices, 'FAMILY_CLUSTER'] = indices.map(sl)
        
        by = ['SENS_CLUSTER','FAMILY_CLUSTER']
        df.dropna(subset=by,inplace=True)
        return df.groupby(by).ngroup()+1