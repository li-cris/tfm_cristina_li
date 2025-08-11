import os
import pickle

import numpy as np
import pandas as pd
import scanpy as sc

import json
from tqdm import tqdm
import os
from collections import Counter, defaultdict
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from torch.utils.data.sampler import Sampler

# FROM SENA
from utils import SCDATA_sampler, SCDataset


class ReplogleDataLoader:
    def __init__(
        self, num_gene_th=5, batch_size=32, dataname="replogle_rpe1_essential", path="data"
    ):
        self.num_gene_th = num_gene_th
        self.batch_size = batch_size
        self.datafile = os.path.join(f"{path}",f"{dataname}", "perturb_processed.h5ad")
        self.ptb_split_path=os.path.join(f"{path}",f"{dataname}", "top_genes.csv")

        # Initialize variables
        self.adata = None
        self.double_adata = None
        self.ptb_targets = None
        self.ptb_targets_affected = None
        self.gos = None
        self.rel_dict = None
        self.gene_go_dict = None
        self.ensembl_genename_mapping_rev = None
        self.gene_var="gene_names"

        # Load the dataset
        self.load_replogle_dataset()

    def load_replogle_dataset(self):
        # Define file path
        fpath = self.datafile

        # Keep only single interventions
        adata = sc.read_h5ad(fpath)
        adata = adata[(~adata.obs["guide_ids"].str.contains(","))]

        # Build gene sets
        gos, GO_to_ensembl_id_assignment, gene_go_dict = self.load_gene_go_assignments(
            adata
        )

        # Compute perturbations with at least 1 gene set
        ptb_targets_affected, _, ensembl_genename_mapping_rev = (
            self.compute_affecting_perturbations(adata, GO_to_ensembl_id_assignment)
        )

        # Build gene-GO relationships
        rel_dict = self.build_gene_go_relationships(
            adata, gos, GO_to_ensembl_id_assignment
        )

        # Load double perturbation data
        ptb_targets = sorted(adata.obs["guide_ids"].unique().tolist())[1:]
        double_adata = sc.read_h5ad(fpath).copy()
        double_adata = double_adata[
            (double_adata.obs["guide_ids"].str.contains(","))
            & (
                double_adata.obs["guide_ids"].map(
                    lambda x: all([y in ptb_targets for y in x.split(",")])
                )
            )
        ]

        # Assign instance variables
        self.adata = adata
        self.double_adata = double_adata
        self.ptb_targets = ptb_targets
        self.ptb_targets_affected = ptb_targets_affected
        self.gos = gos
        self.rel_dict = rel_dict
        self.gene_go_dict = gene_go_dict
        self.ensembl_genename_mapping_rev = ensembl_genename_mapping_rev

    def load_gene_go_assignments(self, adata):
        # load GOs
        GO_to_ensembl_id_assignment = pd.read_csv(
            os.path.join("data", "go_kegg_gene_map.tsv"), sep="\t"
        )
        GO_to_ensembl_id_assignment.columns = ["GO_id", "ensembl_id"]

        # Reduce GOs to the genes we have in adata
        GO_to_ensembl_id_assignment = GO_to_ensembl_id_assignment[
            GO_to_ensembl_id_assignment["ensembl_id"].isin(adata.var_names)
        ]

        # Define GOs and filter
        gos = sorted(
            set(
                pd.read_csv(os.path.join("data", "topGO_uhler.tsv"), sep="\t")[
                    "PathwayID"
                ].values.tolist()
            )
        )
        GO_to_ensembl_id_assignment = GO_to_ensembl_id_assignment[
            GO_to_ensembl_id_assignment["GO_id"].isin(gos)
        ]

        # Keep only gene sets containing more than num_gene_th genes
        counter_genesets_df = pd.DataFrame(
            Counter(GO_to_ensembl_id_assignment["GO_id"]), index=[0]
        ).T
        genesets_in = counter_genesets_df[
            counter_genesets_df.values >= self.num_gene_th
        ].index
        GO_to_ensembl_id_assignment = GO_to_ensembl_id_assignment[
            GO_to_ensembl_id_assignment["GO_id"].isin(genesets_in)
        ]

        # Redefine GOs
        gos = sorted(GO_to_ensembl_id_assignment["GO_id"].unique())

        # Generate gene-GO dictionary
        gene_go_dict = defaultdict(list)
        for go, ens in GO_to_ensembl_id_assignment.values:
            gene_go_dict[ens].append(go)

        return gos, GO_to_ensembl_id_assignment, gene_go_dict

    def compute_affecting_perturbations(self, adata, GO_to_ensembl_id_assignment):
        # Filter interventions not in any GO
        ensembl_genename_mapping = pd.read_csv(
            os.path.join("data", "ensembl_genename_mapping.tsv"), sep="\t"
        )
        ensembl_genename_mapping_dict = dict(
            zip(
                ensembl_genename_mapping.iloc[:, 0], ensembl_genename_mapping.iloc[:, 1]
            )
        )
        ensembl_genename_mapping_rev = dict(
            zip(
                ensembl_genename_mapping.iloc[:, 1], ensembl_genename_mapping.iloc[:, 0]
            )
        )

        # Get intervention targets
        intervention_genenames = map(
            lambda x: ensembl_genename_mapping_dict.get(x, None),
            GO_to_ensembl_id_assignment["ensembl_id"],
        )
        ptb_targets = list(
            set(intervention_genenames).intersection(
                set([x for x in adata.obs["guide_ids"] if x != "" and "," not in x])
            )
        )
        ptb_targets_ens = list(
            map(lambda x: ensembl_genename_mapping_rev[x], ptb_targets)
        )

        return ptb_targets, ptb_targets_ens, ensembl_genename_mapping_rev

    def build_gene_go_relationships(self, adata, gos, GO_to_ensembl_id_assignment):
        # Get genes
        genes = adata.var.index.values
        go_dict = dict(zip(gos, range(len(gos))))
        gen_dict = dict(zip(genes, range(len(genes))))
        rel_dict = defaultdict(list)
        gene_set, go_set = set(genes), set(gos)

        for go, gen in zip(
            GO_to_ensembl_id_assignment["GO_id"],
            GO_to_ensembl_id_assignment["ensembl_id"],
        ):
            if (gen in gene_set) and (go in go_set):
                rel_dict[gen_dict[gen]].append(go_dict[go])

        return rel_dict

    def get_data(self, mode="train"):
        assert mode in ["train", "test"], "mode not supported!"

        if mode == "train":
            dataset = SCDataset(
                adata=self.adata,
                double_adata=self.double_adata,
                ptb_targets=self.ptb_targets,
                perturb_type="single",
            )

            split_ptbs = pd.read_csv(self.ptb_split_path)
            train_idx, test_idx = self.split_scdata(
                dataset,
                split_ptbs=split_ptbs['guide_ids'].tolist(),
            )  # Leave out some cells from the top 12 single target-gene interventions

            ptb_genes = dataset.ptb_targets

            dataset1 = Subset(dataset, train_idx)
            ptb_name = dataset.ptb_names[train_idx]
            dataloader = DataLoader(
                dataset1,
                batch_sampler=SCDATA_sampler(dataset1, self.batch_size, ptb_name),
                num_workers=0,
            )

            dim = dataset[0][0].shape[0]
            cdim = dataset[0][2].shape[0]

            dataset2 = Subset(dataset, test_idx)
            ptb_name = dataset.ptb_names[test_idx]
            dataloader2 = DataLoader(
                dataset2,
                batch_sampler=SCDATA_sampler(dataset2, 8, ptb_name),
                num_workers=0,
            )

            return dataloader, dataloader2, dim, cdim, ptb_genes

        elif mode == "test":
            dataset = SCDataset(
                adata=self.adata,
                double_adata=self.double_adata,
                ptb_targets=self.ptb_targets,
                perturb_type="double",
            )
            ptb_genes = dataset.ptb_targets

            dataloader = DataLoader(
                dataset,
                batch_sampler=SCDATA_sampler(dataset, self.batch_size),
                num_workers=0,
            )

            dim = dataset[0][0].shape[0]
            cdim = dataset[0][2].shape[0]

            return dataloader, dim, cdim, ptb_genes

    def split_scdata(self, scdataset, split_ptbs, pct=0.2):
        # Split data into training and testing
        test_idx = []
        for ptb in split_ptbs:
            idx = np.where(scdataset.ptb_names == ptb)[0]
            test_idx.append(np.random.choice(idx, int(len(idx) * pct), replace=False))
        test_idx = np.hstack(test_idx)
        train_idx = np.array([l for l in range(len(scdataset)) if l not in test_idx])
        return train_idx, test_idx


def check_and_load_paths(data_path: str, model_path: str, ptb_path: str, config_path: str, mode: str, savedir: str) -> None:
    """Check if the given path exists and is a directory."""
    # Loading required data
    if os.path.exists(data_path):
        with open(data_path, "rb") as f:
            dataloader = pickle.load(f)
    else:
        raise FileNotFoundError(f"{mode} data file not found at {data_path}")

    # Loading model
    if os.path.exists(model_path):
        model = torch.load(model_path)
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Loading perturbation file
    if os.path.exists(ptb_path):
        with open(ptb_path, "rb") as f:
            ptb_genes = pickle.load(f)
    else:
        raise FileNotFoundError(f"Perturbation file not found at {ptb_path}")

    # Load config
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        # If config file does not exist, use default values or raise an error
        config = {}
        print(f"Warning: Config file not found in {savedir}. Using default parameters.")
    return(dataloader, model, ptb_genes, config)




def find_pert_pairs(dataloader: str, device: str) -> None:
    """Find perturbation pairs from the dataloader."""
    cidx_list = []
    # Grouping by latent space
    for i, X in enumerate(tqdm(dataloader, desc="Finding intervention pairs")):
        c = X[2].to(device)
        if i == 0:
            c_shape = c

        idx = torch.nonzero(torch.sum(c, axis=0), as_tuple=True)[0]
        idx_pair = idx.cpu()
        cidx_list.append(idx_pair.numpy())

    # Finding unique combinations and index equivalent in whole data
    all_pairs, pair_indices = np.unique(cidx_list, axis=0, return_inverse=True)

    return all_pairs, pair_indices, c_shape