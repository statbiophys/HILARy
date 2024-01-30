import os
import pandas as pd
from Bio import Seq, SeqIO
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def applyParallel(dfGrouped, func, silent=False, cpuCount=None):
    if cpuCount is None: cpuCount = cpu_count()
    with Pool(cpuCount) as p:
        ret_list = list(tqdm(p.imap(func, dfGrouped),total=len(dfGrouped),disable=silent))
    return pd.concat(ret_list)

class Compatible():
    """ Make Briney et al. 2019 data compatible with AIRR schema """
    def __init__(self,
                filename_v=None,filename_j=None,
                offset_v=50,offset_j=33,
                threads=None):
        dirname = os.path.dirname(__file__)
        if filename_v is None: self.filename_v = dirname + '/data/IGHV.fasta'
        else: self.filename_v = filename_v
        if filename_j is None: self.filename_j = dirname + '/data/IGHJ.fasta'
        else: self.filename_j = filename_j

        if threads is None: self.threads = cpu_count()-1
        else: self.threads = threads

        self.genes_v = self.fasta2df(self.filename_v)
        self.genes_j = self.fasta2df(self.filename_j)
        aa103 = 3*(104-1)
        self.genes_v['anchor_index'] =  aa103 - self.genes_v['germline_alignment'].str.count(r'\.')
        self.genes_j['anchor_index'] =  self.genes_j['germline_alignment'].str.find('tgggg')
        self.offset_v = offset_v
        self.offset_j = offset_j
        self.by = ['v_full','j_full','cdr3_length']
        self.columns = ['seq_id',
                        'v_full','j_full','junction',
                        'v_sequence_alignment','j_sequence_alignment',
                        'v_germline_alignment','j_germline_alignment']
        self.rename = {'v_full': 'v_call',
                       'j_full': 'j_call'}

    def fasta2df(self,filename):
        """
        Read an IMGT fasta file, return essential information in a dataframe
        """
        with open(filename) as fasta_file:
            descriptions = []
            seqs = []
            for record in SeqIO.parse(fasta_file, 'fasta'):
                descriptions.append(record.description)
                seqs.append(str(record.seq))
        df = pd.DataFrame(list(zip(descriptions, seqs)), columns =['label', 'germline_alignment'])
        head = ['_']*16
        head[1] = 'call'
        df[head] = df['label'].str.split('|',n=16,expand=True)
        return df[['call','germline_alignment']]

    def group2airr(self,args):
        (v_call,j_call,cdr3_length),df = args
        v = self.genes_v.loc[self.genes_v['call']==v_call]
        j = self.genes_j.loc[self.genes_j['call']==j_call]
        if len(v)==1 and len(j)==1:
            cys104 = v['anchor_index'].values[0]
            try118 = j['anchor_index'].values[0]
            df['N'] = 'N'
            df['NN'] = df.N*df.v_start
            df['offset_v_sequence_alignment'] = (df.NN + df.vdj_nt).str[:cys104+3]
            df['offset_v_germline_alignment'] = v['germline_alignment'].values[0].replace('.','')[:cys104+3]
            df['v_sequence_alignment'] = df['offset_v_sequence_alignment'].str[self.offset_v:]
            df['v_germline_alignment'] = df['offset_v_germline_alignment'].str[self.offset_v:].str.upper()
            df['junction'] = (df.NN + df.vdj_nt).str[cys104:cys104+cdr3_length+6]
            df['j_sequence_alignment'] = (df.NN + df.vdj_nt).str[cys104+cdr3_length+3:].str[:self.offset_j]
            df['j_germline_alignment'] = j['germline_alignment'].values[0][try118:].upper()[:self.offset_j]
            return df[self.columns]
        else: return pd.DataFrame()

    def df2airr(self,df):
        df.dropna(inplace=True)
        df['cdr3_length'] = df['cdr3_nt'].str.len()
        loc = (df['chain']=='heavy')
        loc = loc& (df['productive']=='yes')
        loc = loc& (df['isotype'].str.startswith('IgG'))
        loc = loc& (df['v_start']<self.offset_v)
        return applyParallel(df.loc[loc].groupby(self.by),
                            self.group2airr,
                            silent=True,
                            cpuCount=self.threads).rename(columns=self.rename)
