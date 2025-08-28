import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple
import random

# Torch
import torch

# GEARS
# from gears import PertData

# scGPT module functions
import scgpt as scg
from scgpt.model import TransformerGenerator

from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import set_seed

# Gears (MODIFIED)
from gears_tools.pertdata import PertData

# own scGPT and metric functions
from scgpt_tools.config_loader import model_config_loading, load_pretrained
from data_utils.metrics import compute_kld, MMDLoss
from scgpt_tools.inference import predict, scGPTMetricEvaluator
from scgpt_tools.data import load_dataset, get_gene_vocab


# Global paths for easy changes inside script
FOUNDATION_MODEL_PATH  = './models/scGPT_human'
PREDICT_DOUBLE = True
PREDICT_SINGLE = False
MODEL_DIR_PATH = './models'
RESULT_DIR_PATH = './results'
DATA_DIR_PATH = './data'
SINGLE_TRAIN_ONLY = True

@dataclass
class Options:
    # Command line parameters
    dataset_name: str = "norman_alt"
    project_name: str = "scgpt"
    split: str = "simulation"
    seed: List[int] = field(default_factory=lambda: [42])
    lr: float = 1e-4  # or 1e-4
    batch_size: int = 4 # Recommended was like 32 but not enought GPU
    eval_batch_size: int = 4
    epochs: int = 15
    pool_size: int = 200  # Random set of control samples to use
    top_deg: int = 100

    # Other parameters that are more likely to be changed
    device: torch.device = field(default=None) # Will be set later unless want to set it here
    pretrained_model: bool = True # For now this only works with pretrained True

    pad_token: str = "<pad>"
    special_tokens: List[str] = field(default_factory=list)

    def __post_init__(self):
        # Populate special_tokens after object initialization
        if not self.special_tokens:  # If the list is empty
            self.special_tokens = [self.pad_token, "<cls>", "<eoc>"]

    pad_value: int = 0  # for padding values
    pert_pad_id: int = 0
    include_zero_gene: str = "all"
    max_seq_len: int = 1536

    # Settings for the model
    embsize: int = 256  # embedding dimension
    d_hid: int = 256  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers: int = 6  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead: int = 8  # number of heads in nn.MultiheadAttention
    n_layers_cls: int = 3
    dropout: int = 0  # dropout probability
    use_fast_transformer: bool = True  # whether to use fast transformer

    # logging
    log_interval: int = 100


def load_config(json_path: str) -> None:
    """
    Load configuration from JSON file.
    """
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"JSON config file not found: {json_path}")

def parse_args():
    """
    Parses command-line arguments with optional configuration from a JSON file.

    Configuration priority is as follows:
    Command-line arguments > JSON config file > default values.

    Returns:
        argparse.Namespace: Parsed arguments containing training and dataset parameters.

    Supported arguments:
        --config (str): Path to a JSON configuration file. Used to pre-fill argument defaults.
        --dataset_name (str): Name of the dataset to use (e.g., 'norman_alt').
        --split (str): Split type for the dataset. Must match a split method from `cell-gears`.
        --seed (int, nargs='+'): List of random seeds for reproducibility (e.g., 42 1337).
        --learning_rate (float): Learning rate for training (default: 0.01).
        --batch_size (int): Training batch size (default: 4).
        --eval_batch_size (int): Evaluation batch size (default: 4).
        --epochs (int): Number of training epochs (default: 10).
        --pool_size (int): Number of control samples to randomly pool (default: 200).
    """
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config', type=str, help='Path to config file')

    # Check if JSON config is provided
    pre_args, remaining_args = pre_parser.parse_known_args()
    json_config = {}
    if pre_args.config:
        json_config = load_config(pre_args.config)

    # Main parser with defaults, using values from JSON if provided
    parser = argparse.ArgumentParser(
        parents=[pre_parser],
        description='Arguments taken from command line, JSON config or default.'
    )

    parser.add_argument('--project_name', type=str,
                        default='scgpt',
                        help='Name of the project.')

    parser.add_argument('--dataset_name', type=str,
                        default=json_config.get('dataset_name', 'norman_alt'),
                        help='Exact name of the dataset used.')

    parser.add_argument('--split', type=str,
                        default=json_config.get('split', 'simulation'),
                        help='Type of split for dataset. Function from cell-gears.')

    parser.add_argument('--seed', type=int, nargs='+', default=[42],
                        help='List of seeds to use for reproducibility.')

    parser.add_argument('--learning_rate', type=float,
                        default=json_config.get('learning_rate', 0.01),
                        help='Learning rate for the optimizer.')

    parser.add_argument('--batch_size', type=int,
                        default=json_config.get('batch_size', 4),
                        help='Batch size for training.')

    parser.add_argument('--eval_batch_size', type=int,
                        default=json_config.get('eval_batch_size', 4),
                        help='Batch size for evaluation.')

    parser.add_argument('--epochs', type=int,
                        default=json_config.get('epochs', 10),
                        help='Number of epochs for training.')

    parser.add_argument('--pool_size', type=int,
                        default=200,
                        help='Random set of control samples to use.')

    parser.add_argument('--top_deg', type=int, default=100,
                        help="Number of top differentially expressed genes to evaluate.")
    

    # Parse all arguments (command line overrides JSON/defaults)
    return parser.parse_args(remaining_args)

def clear_track_cache():
    """
    Clear cache for low GPU workspaces. Might not be necessary but it is useful to avoid memory issues.
    This function is called before training and validation.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available on this system.")
        return

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Optional checking of allocated memory
    # Total memory and allocated memory on the GPU
    total_memory = torch.cuda.get_device_properties(0).total_memory  # Total memory of the GPU
    allocated_memory = torch.cuda.memory_allocated(0)  # Memory currently allocated
    reserved_memory = torch.cuda.memory_reserved(0)  # Memory reserved by PyTorch

    # Free memory (total memory - allocated memory)
    free_memory = total_memory - allocated_memory

    # Should return this printed
    print(f"Total memory: {total_memory / 1e9:.2f} GB")
    print(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
    print(f"Reserved memory: {reserved_memory / 1e9:.2f} GB")
    print(f"Free memory: {free_memory / 1e9:.2f} GB")


def load_model(opts: Options,
               vocab,
               model_path: str,
               logger: scg.logger) -> TransformerGenerator:
    """
    Loads a pretrained TransformerGenerator model from a specified checkpoint.

    The model architecture is initialized using the configuration parameters from `opts`.
    It then loads the pretrained weights from the file `best_model.pt` located in `model_path`.

    Args:
        opts (Options): Object containing model configuration parameters.
        vocab (Vocab): Vocabulary object used to determine input/output dimensions.
        model_path (str): Path to the directory containing the pretrained model checkpoint.
        logger (scg.logger): Logger used for recording model loading information.

    Returns:
        TransformerGenerator: The initialized and pretrained model ready for inference or fine-tuning.
    """
    # Model architecture parameters
    embsize = opts.embsize                  # Embedding dimension
    d_hid = opts.d_hid                      # Feedforward network dimension
    nlayers = opts.nlayers                  # Number of encoder layers
    nhead = opts.nhead                      # Number of attention heads
    n_layers_cls = opts.n_layers_cls        # Classification head layers
    dropout = opts.dropout                  # Dropout rate
    use_fast_transformer = opts.use_fast_transformer  # Whether to use a fast transformer variant
    ntokens = len(vocab)                    # Vocabulary size

    # Initialize model with specified configuration
    model = TransformerGenerator(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        n_cls=1,
        nlayers_cls=n_layers_cls,
        vocab=vocab,
        dropout=dropout,
        pad_token=opts.pad_token,
        pad_value=opts.pad_value,
        pert_pad_id=opts.pert_pad_id,
        use_fast_transformer=use_fast_transformer,
    )

    # Load pretrained weights from checkpoint
    scgpt_model_file = os.path.join(model_path, "best_model.pt")
    model = load_pretrained(model, torch.load(scgpt_model_file), logger=logger)

    return model



def main(args):

    clear_track_cache()

    opts = Options(
        dataset_name=args.dataset_name,
        project_name=args.project_name,
        split=args.split,
        seed=args.seed,
        lr=args.learning_rate,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        pool_size=args.pool_size,
        top_deg=args.top_deg
    )

    scgpt_savedir = os.path.join(MODEL_DIR_PATH, opts.project_name)
    for current_seed in opts.seed:
        print(f"Running evaluation with seed {current_seed}")
        # Path settings
        loaded_model_name = f"scgpt_{opts.dataset_name}_{opts.split}_seed_{current_seed}"
        loaded_model_path = os.path.join(scgpt_savedir, loaded_model_name)
        result_savedir = os.path.join(RESULT_DIR_PATH, opts.project_name)
        os.makedirs(result_savedir, exist_ok=True)
        print(f"saving to {result_savedir}")

        # Logger
        logger = scg.logger
        scg.utils.add_file_handler(logger, os.path.join(result_savedir, "predict_evaluate.log"))
        # log running date and current git commit
        logger.info(f"Running on {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Random seed
        set_seed(current_seed)

        # Set device
        if opts.device is not None:
            device = opts.device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        # Load data and model
        pertdata = load_dataset(opts, current_seed, DATA_DIR_PATH, SINGLE_TRAIN_ONLY)
        gene_ids, vocab = get_gene_vocab(pertdata, opts, FOUNDATION_MODEL_PATH)
        model = load_model(opts, vocab, loaded_model_path, logger)
        model.to(device)

        clear_track_cache()

        # Setting up csv file for metrics
        if PREDICT_DOUBLE:
            results_file_path = os.path.join(
                result_savedir, f"{loaded_model_name}_double_metrics.csv")
            prediction_file_path = os.path.join(result_savedir, f"{loaded_model_name}_double.csv")
        elif PREDICT_SINGLE:
            results_file_path = os.path.join(
                result_savedir, f"{loaded_model_name}_single_metrics.csv")
            prediction_file_path = os.path.join(result_savedir, f"{loaded_model_name}_single.csv")

        # Setting up evaluator
        evaluator = scGPTMetricEvaluator(opts, gene_ids, current_seed, results_file_path,
                                         model, pertdata)

        # Evaluating
        if PREDICT_DOUBLE:
            mean_result_pred = evaluator.evaluate_double()

        elif PREDICT_SINGLE:
            if 'split_dict' not in pertdata.adata.obs.columns:
                temp_data_path = os.path.join(DATA_DIR_PATH, args.dataset_name, 'split_columns')
                split_column_file = f"{args.dataset_name}_split_{args.split}_seed_{str(current_seed)}_split_column.csv"
                split_column = pd.read_csv(os.path.join(temp_data_path, split_column_file), index_col=0)

                pertdata.adata.obs['split_dict'] = split_column['split']

            mean_result_pred = evaluator.evaluate_single()
        else:
            print("Evaluation not set. Stopping evaluation.")
            return

        # Saving predictions
        predictions_df = pd.DataFrame.from_dict(mean_result_pred, orient='index')
        predictions_df.columns = pertdata.adata.var_names
        # Turn index into column
        predictions_df.reset_index(inplace=True, names='double')

        # Save predictions
        predictions_df.to_csv(prediction_file_path, index=False)
        print(f"Mean predictions saved to {prediction_file_path}.")

        clear_track_cache()



if __name__ == '__main__':
    args = parse_args()
    main(args)