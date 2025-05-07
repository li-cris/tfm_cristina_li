import scanpy as sc
import numpy as np
from gears import PertData

import os
import sys
sys.path.insert(0, os.path.abspath('../'))
from cris_test.single_norman_utils import separate_data, get_common_genes

MODEL_DIR_PATH = '../cris_test/models/'
RESULTS_DIR_PATH = '../cris_test/results/'
DATA_DIR_PATH = '../cris_test/data/'

adata_filepath = '/wdir/tfm/SENA/data/Norman2019_raw.h5ad'
adata = sc.read(adata_filepath)

adata.var['gene_name'] = adata.var['gene_symbols']

# pert_data = PertData('../cris_test/data') # specific saved folder
# pert_data.load(data_name = 'norman') # specific dataset name

print("Changing Norman adata based on lgem...")
# Changing adata to keep only features common with 
# alt_adata = pert_data.adata

# Changing adata to keep only features common with 
adata_single, adata_double, adata_ctrl = separate_data(adata = adata, dataset_name = 'norman') # get single, double and control datasets

adata_both = adata_single.concatenate(adata_double, join = 'outer', index_unique = '-')
all_perts, perts, genes, adata_common = get_common_genes(adata = adata_both, dataset_name = "norman")

norman_alt = adata_common.concatenate(adata_ctrl, join = 'outer', index_unique = '-')
shuffled_indices = np.random.permutation(norman_alt.n_obs)
alt_adata = norman_alt[shuffled_indices].copy()

# Turn single back into GEARS accepted naming style

pert_data = PertData('../cris_test/data') # specific saved folder
pert_data.new_data_process(dataset_name = 'norman_alt', adata = alt_adata) # specific dataset name and adata object
pert_data.load(data_path = '../cris_test/data/norman_alt') # load the processed data, the path is saved folder + dataset_name
pert_data.prepare_split(split = 'simulation', seed = 42) # get data split with seed
pert_data.get_dataloader(batch_size = 32, test_batch_size = 128) # prepare data loader