import numpy as np
import torch


def separate_data(adata = None, dataset_name: str = "norman"):
    """Get the single perturbation dataset, double perturbation dataset and control dataset from the given AnnData object as well as list of single perturbations."""
    if "norman" in dataset_name.lower():
       # Preprocessing dataset
        splitting = adata.obs['condition'].str.split('+')
        for i in range(len(splitting)):
            if len(splitting[i]) == 2:
                if 'ctrl' in splitting[i]:
                    splitting[i].remove('ctrl')

        join_names = splitting.apply(lambda x: '+'.join(sorted(x))) # Makes sure that order is the same
        adata.obs['condition_fixed'] = join_names

        # Keeping only single perturbations
        filter_mask = ~adata.obs["condition_fixed"].str.contains(r"\+") # mask for those NOT containing +
        indexes_to_keep = filter_mask[filter_mask].index # mask that finds indeces in norman adata that aren't double perturbations

        # Dataset with single perts
        adata_single = adata[indexes_to_keep].copy()
        adata_single = adata_single[adata_single.obs['condition_fixed']!='ctrl']

        # Dataset with double perts
        adata_double = adata[~adata.obs['condition_fixed'].isin(adata_single.obs['condition_fixed'])].copy()
        adata_double = adata_double[adata_double.obs['condition_fixed']!='ctrl']

        # Ctrl expression
        adata_ctrl = adata[adata.obs['condition_fixed']=='ctrl'].copy()


    else:
        print("Dataset not implemented yet.")

    return adata_single, adata_double, adata_ctrl


def get_common_genes(adata = None, dataset_name: str = "norman"):
    """Get adata with perts found in genes (features) and return list of perts and genes."""
    if "norman" in dataset_name.lower():
        all_perts = adata.obs['condition_fixed'].values
        genes = set(adata.var['gene_symbols'].values)

        # Makes function work even with double perts
        def valid_pert(pert):
            pair_genes = pert.split("+")
            return all(gene in genes for gene in pair_genes)

        valid_perts = [p for p in all_perts if valid_pert(p)]
        adata_common = adata[adata.obs["condition_fixed"].isin(valid_perts)].copy() # Only keep perts that are in features
        perts = adata_common.obs["condition_fixed"].unique().tolist()
    else:
        print("Dataset not implemented yet.")
    return all_perts, perts, list(genes), adata_common