import os
import numpy as np
from gears import PertData
from scgpt.tokenizer.gene_tokenizer import GeneVocab


def load_dataset(opts, DATA_DIR_PATH: str) -> PertData:
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
    seed = opts.seed # Review, might change with iterations
    batch_size = opts.batch_size
    eval_batch_size = opts.eval_batch_size

    # Loading dataset from directory it is found in
    pert_data = PertData(DATA_DIR_PATH)
    pert_data.load(data_path=os.path.join(DATA_DIR_PATH, dataset_name))
    pert_data.prepare_split(split=split, seed=seed)
    pert_data.get_dataloader(batch_size=batch_size, test_batch_size=eval_batch_size)

    return pert_data

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