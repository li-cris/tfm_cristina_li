import scanpy as sc
import numpy as np
from gears import PertData
import os

# Global paths
DATASET_NAME = 'replogle' # For now only norman dataset is implemented
DATA_DIR_PATH = '../cris_test/data/' # Directory where new Norman dataset will be stored

# Path to Norman dataset
DATASET_PATH = '/wdir/tfm/cris_test/data/replogle_rpe1_essential/perturb_processed.h5ad'
# or '../SENA/data/Norman2019_raw.h5ad'
NEW_DATASET_NAME = 'replogle_rpe1_alt' # new name for processed dataset (or norman_alt)

def separate_data(adata = None, dataset_name = "norman"):
    """
    Get the single perturbation dataset, double perturbation dataset and control dataset from the given AnnData object as well as list of single perturbations.
    Review this: Could be redundant since GEARS data handler can already separate control, singles and doubles.
    """
    if dataset_name == "norman" or dataset_name == "replogle":
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


def get_common_genes(adata = None, dataset_name = "norman"):
    """
    Get adata with perts found in genes (features) and return list of perts and genes.
    """
    if dataset_name == "norman" or dataset_name == "replogle":
        all_perts = adata.obs['condition_fixed'].values
        genes = set(adata.var['gene_symbols'].values)

        # To consider double perts too
        def valid_pert(pert):
            pair_genes = pert.split("+")
            return all(gene in genes for gene in pair_genes)

        valid_perts = [p for p in all_perts if valid_pert(p)]
        adata_common = adata[adata.obs["condition_fixed"].isin(valid_perts)].copy() # Only keep perts that are in features
        perts = adata_common.obs["condition_fixed"].unique().tolist()

    else:
        print("Dataset not implemented yet.")

    return all_perts, perts, list(genes), adata_common


def main():

    # Read AnnData (.h5ad) file
    adata = sc.read(DATASET_PATH)

    if DATASET_NAME == 'norman' or DATASET_NAME == 'replogle':
        # GEARS accepted naming style
        adata.var['gene_name'] = adata.var['gene_symbols']

    # Separate data into single and double perturbations and control
    adata_single, adata_double, adata_ctrl = separate_data(adata = adata, dataset_name = DATASET_NAME) # get single, double and control datasets

    # Join single and double perturbations (could be done by filtering out 'ctrl' from adata)
    adata_both = adata_single.concatenate(adata_double, join = 'outer', index_unique = '-')
    all_perts, perts, genes, adata_common = get_common_genes(adata = adata_both, dataset_name = DATASET_NAME)

    # Join new filtered pertrubation data with control samples
    adata_filtered = adata_common.concatenate(adata_ctrl, join = 'outer', index_unique = '-')
    # Reshuffle samples
    shuffled_indices = np.random.permutation(adata_filtered.n_obs)
    alt_adata = adata_filtered[shuffled_indices].copy()

    # Creating PertData object for new dataset
    # .h5ad file is saved in data folder as perturb_processed.h5ad
    pert_data = PertData(DATA_DIR_PATH) # specific saved folder
    pert_data.new_data_process(dataset_name = NEW_DATASET_NAME, adata = alt_adata) # specific dataset name and adata object
    pert_data.load(data_path = os.path.join(DATA_DIR_PATH, NEW_DATASET_NAME)) # load the processed data, the path is saved folder + dataset_name

    print('Dataset created and loaded successfully. Attempting to prepare split and dataloader...')
    pert_data.prepare_split(split = 'simulation', seed = 42) # get data split with seed
    pert_data.get_dataloader(batch_size = 32, test_batch_size = 128) # prepare data loader


if __name__ == "__main__":
    main()