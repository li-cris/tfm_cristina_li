import os
import numpy as np
import pandas as pd
from gears import PertData
from scgpt.tokenizer.gene_tokenizer import GeneVocab


def load_dataset(opts, seed: int, DATA_DIR_PATH: str, SINGLE_DATA_ONLY: bool = True) -> PertData:
    """
    Loads and prepares the perturbation dataset for training and evaluation.

    This function initializes a PertData object from the specified directory,
    applies dataset splitting, and creates data loaders for training and testing.

    Args:
        opts: Options object containing dataset and batch configuration.
        DATA_DIR_PATH (str): Root directory path where datasets are stored.

    Returns:
        PertData: An object containing the loaded dataset, splits, and dataloaders
                  ready for use in training and validation loops.
    """
    # Specifying parameters
    dataset_name = opts.dataset_name
    split = opts.split
    batch_size = opts.batch_size
    eval_batch_size = opts.eval_batch_size

    # Loading dataset from directory it is found in
    pertdata = PertData(DATA_DIR_PATH)
    pertdata.load(data_path=os.path.join(DATA_DIR_PATH, dataset_name))


    if split == 'simulation_single':
        print("Training and evaluating with single perturbation data.")
        pertdata.adata = pertdata.adata[pertdata.adata.obs['condition'].str.contains('ctrl')]
        pertdata.prepare_split(split=split, seed=seed)

        temp_data_path = os.path.join(DATA_DIR_PATH, dataset_name, 'split_columns')
        if not os.path.exists(temp_data_path):
            os.makedirs(temp_data_path, exist_ok=True)
        split_column_file = f"{dataset_name}_split_{split}_seed_{str(seed)}_split_column.csv"

        # Saving the created adata.obs['split] because it doesn't get saved in pertdata.adata when reloading
        if 'split' in pertdata.adata.obs.columns:
            pertdata.adata.obs['split'].to_csv(os.path.join(temp_data_path, split_column_file), index=True)

        # Readding the split column if it was not present in adata.obs for later use
        if 'split' not in pertdata.adata.obs.columns:
            split_column = pd.read_csv(os.path.join(temp_data_path, split_column_file), index_col=0)
            pertdata.adata.obs['split'] = split_column['split']


    elif SINGLE_DATA_ONLY and split != 'simulation_single':
        print("Keeping only single perturbation samples in training.")
        pertdata.prepare_split(
            split=split,
            seed=seed,
            train_gene_set_size=1.0,
            combo_seen2_train_frac=0.0
        )

    else:
        print("Training with both single and double perturbation data and evaluating with double perturbations.")
        pertdata.prepare_split(split=split, seed=seed)

    pertdata.get_dataloader(batch_size=batch_size, test_batch_size=eval_batch_size)

    return pertdata


def get_gene_vocab(pert_data: PertData, opts: None, foundation_model_path: str) -> None:
    """
    Loads the gene vocabulary from a pretrained foundation model and maps dataset genes to IDs.

    This function reads the vocabulary JSON file from the foundation model directory,
    ensures special tokens are included, sets the default padding token, and returns
    the gene IDs corresponding to the dataset's gene list.

    Args:
        pert_data (PertData): Dataset object containing gene expression data.
        opts (Options): Configuration object including special tokens.
        foundation_model_path (str): Path to the pretrained foundation model directory.

    Returns:
        Tuple[np.ndarray, GeneVocab]: 
            - gene_ids: Array of gene indices mapped from the vocabulary.
            - vocab: Loaded and possibly extended GeneVocab object.
    """

    # Load vocab for gene_ids
    vocab_file = os.path.join(foundation_model_path, "vocab.json")
    vocab = GeneVocab.from_file(vocab_file)
    for s in opts.special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    vocab.set_default_index(vocab["<pad>"])
    genes = pert_data.adata.var["gene_name"].tolist()
    gene_ids = np.array(
        [vocab[gene] if gene in vocab else vocab["<pad>"] for gene in genes], dtype=int
        )

    return gene_ids, vocab