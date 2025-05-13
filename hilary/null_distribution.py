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
