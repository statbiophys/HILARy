import os
from typing import Iterable

import numpy as np
import pandas as pd
import righor
import sonnia
from sonnia.sonia import Sonia
from sonnia.sonnia import SoNNia
from tqdm import tqdm

from hilary.apriori import Apriori
from hilary.generate_conditional import generate_pgen_seqs_righor
from hilary.utils import create_classes


class NullDistribution:
    """
    Class to generate null distributions of CDR3s distances.
    It uses the righor+SoNNia for generating synthetic sequences and hilary for calculating distance histograms.
    """
    def __init__(self, n_seqs=int(1e7), ppost_model='human_B_heavy',linear=True):
        """
        Initialize the NullDistribution class.
        Parameters:
            n_seqs (int): Number of sequences to generate.
            ppost_model (str): Directory containing the model files.
            linear (bool): Whether to use the linear model or not.
        """
        self.n_seqs = n_seqs
        # load sonia model
        if linear:
            self.sonia_model=Sonia(ppost_model=ppost_model)
        else:
            self.sonia_model=SoNNia(ppost_model=ppost_model)

        # load righor model
        main_folder = os.path.dirname(sonnia.__file__)
        default_models=os.listdir(os.path.join(main_folder, "default_models"))
        if ppost_model in default_models:
            model_directory = os.path.join(main_folder, "default_models", ppost_model)
            print('Loading default model from',model_directory)
        else:
            model_directory = ppost_model
        self.righor_model=righor.load_model_from_files(None,os.path.join(model_directory,'model_params.txt'),
                                                os.path.join(model_directory,'model_marginals.txt'),
                                                os.path.join(model_directory,'V_gene_CDR3_anchors.csv'),
                                                os.path.join(model_directory,'J_gene_CDR3_anchors.csv'))
        self.v_genes=np.unique([v.split('*')[0] for v in self.sonia_model.pgen_model.V_allele_names])
        self.j_genes=np.unique([v.split('*')[0] for v in self.sonia_model.pgen_model.J_allele_names])
        self.get_common_pairs()

    def get_common_pairs(self,
                         upper_bound: int = 15,
                         n_seqs: int = int(1e6),
                         max_n_vjs: int = 150) -> None:
        """
        Get common VJ pairs from the generated sequences.
        Parameters:
            upper_bound (float): Upper bound for the selection factor.
            n_seqs (int): Number of sequences to generate.
            max_n_vjs (int): Maximum number of VJ pairs to keep.
        """
        # generate sequences
        pgen_seqs=generate_pgen_seqs_righor(self.righor_model,int(n_seqs*upper_bound*1.1))
        Qs=self.sonia_model.evaluate_selection_factors(pgen_seqs[['junction_aa','v_gene','j_gene']].values)
        random_samples = np.random.uniform(size=len(Qs))
        selection=random_samples < Qs / upper_bound
        # select ppost sequences
        post_seqs=pgen_seqs[selection].reset_index(drop=True)
        # count the number of sequences for each VJ pair
        vjl_classes=post_seqs.groupby(['v_gene','j_gene']).size().reset_index().rename(columns={0:'count'})
        # sort the VJ pairs by count
        vjl_classes.sort_values('count',ascending=False,inplace=True)
        # get the top VJ pairs
        vjl_classes=vjl_classes.loc[vjl_classes['count']>10]
        self.common_vjs=vjl_classes[['v_gene','j_gene']].values
        print('Found',len(self.common_vjs),'vj pairs with more than 10 counts')
        if len(self.common_vjs)>max_n_vjs:
            self.common_vjs=self.common_vjs[:max_n_vjs]
            print('Reduced to',len(self.common_vjs),'vj pairs')

    def all_nulls(self,
                  all: bool = False) -> pd.DataFrame:
        """
        Generate all null distributions for the common VJ pairs.
        Parameters:
            all (bool): Whether to generate all VJ pairs or only the common ones.
        Returns:
            pd.DataFrame: DataFrame containing the null distributions.
        """
        self.null=pd.DataFrame()
        if all:
            for j_gene in self.j_genes:
                for v_gene in self.v_genes:
                    vj_null= self.build_null_distribution(n_seqs=self.n_seqs, available_j=[j_gene], available_v=[v_gene])
                    self.null=pd.concat([self.null,vj_null],ignore_index=True).reset_index(drop=True)
        else:
            for v_gene,j_gene in tqdm(self.common_vjs):
                vj_null= self.build_null_distribution(n_seqs=self.n_seqs, available_j=[j_gene], available_v=[v_gene])
                self.null=pd.concat([self.null,vj_null],ignore_index=True).reset_index(drop=True)

        jl=self.null.drop(columns=['v_gene']).groupby(['j_gene','cdr3_length']).mean().reset_index()
        l=self.null.drop(columns=['v_gene','j_gene']).groupby(['cdr3_length']).mean().reset_index()
        self.all=pd.concat([self.null,jl,l])
        return self.all

    def build_null_distribution(self,
                                n_seqs: int | None = None,
                                available_j: Iterable | None = None,
                                available_v: Iterable | None = None,
                                sel_counts: int = 300,
                                upper_bound: int = 15,
                                seed:int = 42) -> pd.DataFrame:
        """
        Build a null distribution for the given VJ pair.
        Parameters:
            n_seqs (int): Number of sequences to generate.
            available_j (list): List of available J genes.
            available_v (list): List of available V genes.
            sel_counts (int): Minimum counts for the null distribution.
            upper_bound (float): Upper bound for the selection factor.
            seed (int): Random seed for reproducibility.
        Returns:
            pd.DataFrame: DataFrame containing the null distribution.
        """
        print('Building null distribution for vj:',available_v,available_j)
        # generate sequences
        seqs=generate_pgen_seqs_righor(self.righor_model,n_seqs,available_j=available_j,available_v=available_v)
        seqs['cdr3']=seqs['junction'].apply(lambda x: x[3:-3])
        seqs['cdr3_length']=seqs['cdr3'].apply(len)
        # infer selection factors
        Qs=self.sonia_model.evaluate_selection_factors(seqs[['junction_aa','v_gene','j_gene']].values)
        mean_Qs = np.mean(Qs) + 1e-9
        apriori_synths,histograms_synths=[],[]
        rng = np.random.default_rng(seed)

        for _ in range(3):
            # select sequences
            random_samples = rng.uniform(size=len(Qs))
            selection=random_samples < Qs / (upper_bound * mean_Qs)
            print('Selected:',selection.sum(),'at a ratio', selection.sum()/len(selection))
            post_seqs=seqs[selection].reset_index(drop=True)
            # calculate histograms
            apriori_synth = Apriori(silent=False, threads=-1, precision=0.99, sensitivity=0.9)
            apriori_synth.classes = create_classes(post_seqs)
            apriori_synths.append(apriori_synth.classes)
            histograms_synths.append(apriori_synth.get_histograms(post_seqs))

        out_dfs=[]
        for hist_,classes in zip(histograms_synths,apriori_synths):
            # calculate histograms
            cumulative_distributions=hist_.values[:,1:].cumsum(axis=1)
            cumulative_distributions=cumulative_distributions/cumulative_distributions[:,-1][:, np.newaxis]
            cumulative_distributions=pd.DataFrame(cumulative_distributions,columns=hist_.columns[1:])
            cumulative_distributions['class_id']=hist_.class_id.values
            # select only the classes with counts > sel_counts
            counts=hist_.values[:,1:].sum(axis=1)
            selected_cumulative=cumulative_distributions[counts>sel_counts]
            merged=classes[['v_gene','j_gene','cdr3_length','class_id']].merge(selected_cumulative)
            out_dfs.append(merged.drop(columns='class_id'))

        return pd.concat(out_dfs).groupby(['v_gene','j_gene','cdr3_length']).mean().reset_index()

class NullDistributionPaired:
    """
    Class to generate null distributions of CDR3s distances.
    It uses the righor+SoNNia for generating synthetic sequences and hilary for calculating distance histograms.
    """
    def __init__(self, n_seqs=int(1e7), heavy_model='human_B_heavy',light_model_1='human_B_kappa',light_model_2='human_B_lambda',linear_heavy=True,linear_light=True):
        """
        Initialize the NullDistributionPaired class.
        Parameters:
            n_seqs (int): Number of sequences to generate.
            heavy_model (str): Directory containing the heavy model files.
            light_model_1 (str): Directory containing the light model files.
            light_model_2 (str): Directory containing the light model files.
            linear (bool): Whether to use the linear model or not.
        """
        self.n_seqs = n_seqs
        # load sonia model
        if linear_heavy:
            self.sonia_model_heavy=Sonia(ppost_model=heavy_model)
        else:
            self.sonia_model_heavy=SoNNia(ppost_model=heavy_model)
        if linear_light:
            self.sonia_model_light_1=Sonia(ppost_model=light_model_1)
            self.sonia_model_light_2=Sonia(ppost_model=light_model_2)
        else:
            self.sonia_model_light_1=SoNNia(ppost_model=light_model_1)
            self.sonia_model_light_2=SoNNia(ppost_model=light_model_2)

        # load righor model
        main_folder = os.path.dirname(sonnia.__file__)
        default_models=os.listdir(os.path.join(main_folder, "default_models"))
        
        
        if heavy_model in default_models:
            model_directory = os.path.join(main_folder, "default_models", heavy_model   )
            print('Loading default model from',model_directory)
        else:
            model_directory = heavy_model
        self.righor_model_heavy=righor.load_model_from_files(None,os.path.join(model_directory,'model_params.txt'),
                                                os.path.join(model_directory,'model_marginals.txt'),
                                                os.path.join(model_directory,'V_gene_CDR3_anchors.csv'),
                                                os.path.join(model_directory,'J_gene_CDR3_anchors.csv'))
        self.v_genes_heavy=np.unique([v.split('*')[0] for v in self.sonia_model_heavy.pgen_model.V_allele_names])
        self.j_genes_heavy=np.unique([v.split('*')[0] for v in self.sonia_model_heavy.pgen_model.J_allele_names])
        
        if light_model_1 in default_models:
            model_directory = os.path.join(main_folder, "default_models", light_model_1)
            print('Loading default model from',model_directory)
        else:
            model_directory = light_model_1
        self.righor_model_light_1=righor.load_model_from_files(None,os.path.join(model_directory,'model_params.txt'),
                                                os.path.join(model_directory,'model_marginals.txt'),
                                                os.path.join(model_directory,'V_gene_CDR3_anchors.csv'),
                                                os.path.join(model_directory,'J_gene_CDR3_anchors.csv'))
        self.v_genes_light_1=np.unique([v.split('*')[0] for v in self.sonia_model_light_1.pgen_model.V_allele_names])
        self.j_genes_light_1=np.unique([v.split('*')[0] for v in self.sonia_model_light_1.pgen_model.J_allele_names])
        
        if light_model_2 in default_models:
            model_directory = os.path.join(main_folder, "default_models", light_model_2)
            print('Loading default model from',model_directory)
        else:
            model_directory = light_model_2
        self.righor_model_light_2=righor.load_model_from_files(None,os.path.join(model_directory,'model_params.txt'),
                                                os.path.join(model_directory,'model_marginals.txt'),
                                                os.path.join(model_directory,'V_gene_CDR3_anchors.csv'),
                                                os.path.join(model_directory,'J_gene_CDR3_anchors.csv'))
        self.v_genes_light_2=np.unique([v.split('*')[0] for v in self.sonia_model_light_2.pgen_model.V_allele_names])
        self.j_genes_light_2=np.unique([v.split('*')[0] for v in self.sonia_model_light_2.pgen_model.J_allele_names])
        
        self.get_common_pairs()
        
    def get_common_pairs(self,
                         upper_bound: int = 15,
                         n_seqs: int| None = None,
                         max_n_vjs: int = 150) -> None:
        """
        Get common J pairs from the generated sequences.
        Parameters:
            upper_bound (float): Upper bound for the selection factor.
            n_seqs (int): Number of sequences to generate.
            max_n_vjs (int): Maximum number of VJ pairs to keep.
        """
        if n_seqs is None:
            n_seqs=self.n_seqs
        # generate sequences
        pgen_seqs=generate_pgen_seqs_righor(self.righor_model_heavy,int(n_seqs*upper_bound*1.1))
        Qs=self.sonia_model_heavy.evaluate_selection_factors(pgen_seqs[['junction_aa','v_gene','j_gene']].values)
        random_samples = np.random.uniform(size=len(Qs))
        selection=random_samples < Qs / upper_bound
        # select ppost sequences
        post_seqs_heavy=pgen_seqs[selection].reset_index(drop=True)
        
        pgen_seqs=generate_pgen_seqs_righor(self.righor_model_light_1,int(n_seqs*upper_bound*1.1/2))
        Qs=self.sonia_model_light_1.evaluate_selection_factors(pgen_seqs[['junction_aa','v_gene','j_gene']].values)
        random_samples = np.random.uniform(size=len(Qs))
        selection=random_samples < Qs / upper_bound
        # select ppost sequences
        post_seqs_light_1=pgen_seqs[selection].reset_index(drop=True)
        
        pgen_seqs=generate_pgen_seqs_righor(self.righor_model_light_2,int(n_seqs*upper_bound*1.1/2))
        Qs=self.sonia_model_light_2.evaluate_selection_factors(pgen_seqs[['junction_aa','v_gene','j_gene']].values)
        random_samples = np.random.uniform(size=len(Qs))
        selection=random_samples < Qs / upper_bound
        # select ppost sequences
        post_seqs_light_2=pgen_seqs[selection].reset_index(drop=True)
        
        ppost_seqs_light=pd.concat([post_seqs_light_1,post_seqs_light_2])
        
        min_length=min(len(post_seqs_heavy),len(ppost_seqs_light))
        post_seqs_heavy=post_seqs_heavy.sample(min_length)
        post_seqs_heavy['seq_id']=np.arange(len(post_seqs_heavy))
        ppost_seqs_light=ppost_seqs_light.sample(min_length)
        ppost_seqs_light['seq_id']=np.arange(len(ppost_seqs_light))
        merged=ppost_seqs_light.merge(post_seqs_heavy,on='seq_id',suffixes=('_light','_heavy'))
        
        # count the number of sequences for each VJ pair
        vjl_classes=merged.groupby(['v_gene_heavy','v_gene_light','j_gene_heavy','j_gene_light']).size().reset_index().rename(columns={0:'count'})
        # sort the VJ pairs by count
        vjl_classes.sort_values('count',ascending=False,inplace=True)
        # get the top VJ pairs
        vjl_classes=vjl_classes.loc[vjl_classes['count']>10]

        print('initial number of vj heavy-light pairs:',len(vjl_classes))
        vjs1=vjl_classes[['v_gene_heavy','v_gene_light','j_gene_heavy','j_gene_light']].groupby(['j_gene_heavy','j_gene_light']).sample(n=3,replace=True).reset_index(drop=True).drop_duplicates()
        vjs2=vjl_classes[['v_gene_heavy','v_gene_light','j_gene_heavy','j_gene_light']][:30]
        self.common_vjs=pd.concat([vjs2,vjs1.sample(n=50,replace=True)]).drop_duplicates().values
        if len(self.common_vjs)>max_n_vjs:
            self.common_vjs=self.common_vjs[:max_n_vjs]
            print('Reduced to',len(self.common_vjs),'vj heavy-light pairs')

    def all_nulls(self,
                  n_seqs:int=int(1e7)) -> pd.DataFrame:
        """
        Generate all null distributions for the common VJ pairs.
        Parameters:
            all (bool): Whether to generate all VJ pairs or only the common ones.
        Returns:
            pd.DataFrame: DataFrame containing the null distributions.
        """
        self.null=pd.DataFrame()
        for v_gene_heavy,v_gene_light,j_gene_heavy,j_gene_light in tqdm(self.common_vjs):
            j_null= self.build_null_distribution(n_seqs=n_seqs,available_v_heavy=[v_gene_heavy], available_v_light=[v_gene_light], available_j_heavy=[j_gene_heavy], available_j_light=[j_gene_light])
            self.null=pd.concat([self.null,j_null],ignore_index=True).reset_index(drop=True)
        jl_null=self.null.drop(columns=['v_gene']).groupby(['j_gene','cdr3_length']).mean().reset_index()
        l=jl_null.drop(columns=['j_gene']).groupby(['cdr3_length']).mean().reset_index()
        self.all=pd.concat([self.null,jl_null,l])
        return self.all

    def build_null_distribution(self,
                                n_seqs: int | None = None,
                                available_j_heavy: Iterable | None = None,
                                available_j_light: Iterable | None = None,
                                available_v_heavy: Iterable | None = None,
                                available_v_light: Iterable | None = None,
                                sel_counts: int = 300,
                                upper_bound: int = 15,
                                seed:int = 42) -> pd.DataFrame:
        """
        Build a null distribution for the given JJ pair.
        Parameters:
            n_seqs (int): Number of sequences to generate.
            available_j_heavy (list): List of available J heavy genes.
            available_j_light (list): List of available J light genes.
            sel_counts (int): Minimum counts for the null distribution.
            upper_bound (float): Upper bound for the selection factor.
            seed (int): Random seed for reproducibility.
        Returns:
            pd.DataFrame: DataFrame containing the null distribution.
        """
        print('Building null distribution for vj:',available_v_heavy,available_v_light,available_j_heavy,available_j_light)
        # generate sequences
        seqs=generate_pgen_seqs_righor(self.righor_model_heavy,n_seqs,available_j=available_j_heavy,available_v=available_v_heavy)
        seqs['cdr3']=seqs['junction'].apply(lambda x: x[3:-3])
        seqs['cdr3_length']=seqs['cdr3'].apply(len)
        Qs_heavy=self.sonia_model_heavy.evaluate_selection_factors(seqs[['junction_aa','v_gene','j_gene']].values)
        
        if 'K' in available_j_light[0].split('-')[0]:
            seqs_light=generate_pgen_seqs_righor(self.righor_model_light_1,n_seqs,available_j=available_j_light,available_v=available_v_light)
        else:
            seqs_light=generate_pgen_seqs_righor(self.righor_model_light_2,n_seqs,available_j=available_j_light,available_v=available_v_light)
        seqs_light['cdr3']=seqs_light['junction'].apply(lambda x: x[3:-3])
        seqs_light['cdr3_length']=seqs_light['cdr3'].apply(len)
        if 'K' in available_j_light[0].split('-')[0]:
            Qs_light=self.sonia_model_light_1.evaluate_selection_factors(seqs_light[['junction_aa','v_gene','j_gene']].values)
        else:
            Qs_light=self.sonia_model_light_2.evaluate_selection_factors(seqs_light[['junction_aa','v_gene','j_gene']].values)
        
        Qs=Qs_heavy*Qs_light
        mean_Qs = np.mean(Qs) + 1e-9
        
        apriori_synths,histograms_synths=[],[]
        rng = np.random.default_rng(seed)

            # select sequences
        random_samples = rng.uniform(size=len(Qs))
        selection=random_samples < Qs / (upper_bound * mean_Qs)
        print('Selected:',selection.sum(),'at a ratio', selection.sum()/len(selection))
        
        post_seqs=seqs[selection].reset_index(drop=True)
        post_seqs_light=seqs_light[selection].reset_index(drop=True)
        min_length=np.min([len(post_seqs),len(post_seqs_light)])
        
        post_seqs=post_seqs.sample(min_length).reset_index(drop=True)
        post_seqs_light=post_seqs_light.sample(min_length).reset_index(drop=True)
        post_seqs['seq_id']=np.arange(len(post_seqs))
        post_seqs_light['seq_id']=np.arange(len(post_seqs_light))
        usecols = [ "seq_id","v_gene", "j_gene", "cdr3_length", "cdr3"]
        df=post_seqs.merge(post_seqs_light, on=['seq_id'],suffixes=('_heavy','_light'))
        for column in usecols[1:]:
            df[column]=df[column+'_heavy'].astype(str)+','+df[column+'_light'].astype(str)
        df=df[usecols]
        # calculate histograms
        apriori_synth = Apriori(silent=False, threads=-1, precision=0.99, sensitivity=0.9,paired=True)
        apriori_synth.classes = create_classes(df)
        hist_=apriori_synth.get_histograms(df)
       
            # calculate histograms
        cumulative_distributions=hist_.values[:,1:].cumsum(axis=1)
        cumulative_distributions=cumulative_distributions/cumulative_distributions[:,-1][:, np.newaxis]
        cumulative_distributions=pd.DataFrame(cumulative_distributions,columns=hist_.columns[1:])
        cumulative_distributions['class_id']=hist_.class_id.values
        # select only the classes with counts > sel_counts
        counts=hist_.values[:,1:].sum(axis=1)
        selected_cumulative=cumulative_distributions[counts>sel_counts]
        merged=apriori_synth.classes[['v_gene','j_gene','cdr3_length','class_id']].merge(selected_cumulative)
        merged['cdr3_length']=merged['cdr3_length'].apply(lambda x: int(x.split(',')[0])+int(x.split(',')[1]))
        self.out_dfs=merged.drop(columns='class_id')
        return self.out_dfs.groupby(['v_gene','j_gene','cdr3_length']).mean().reset_index()