# General modules
import argparse
import json
import os
import time
import copy
from typing import List, Tuple
from dataclasses import dataclass, field

# PyTorch
import torch
from torch import nn

# GEARS
# from gears import PertData


# scGPT module functions
import scgpt as scg
from scgpt.model import TransformerGenerator
from scgpt.loss import (
    masked_mse_loss,
    criterion_neg_log_bernoulli,
    masked_relative_error
)
from scgpt.tokenizer import tokenize_batch, pad_batch, tokenize_and_pad_batch
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import set_seed, compute_perturbation_metrics

# GEARS
from gears_tools.pertdata import PertData

# own scGPT functions
from scgpt_tools.config_loader import model_config_loading, load_pretrained
from scgpt_tools.model import train, eval_perturb, validate_perturbation_model
from scgpt_tools.data import load_dataset

# Set up directories (editable mode)
FOUNDATION_MODEL_PATH  = './models/scGPT_human'
MODEL_DIR_PATH = './models'
RESULT_DIR_PATH = './results'
DATA_DIR_PATH = './data'
SINGLE_TRAIN_ONLY = True

@dataclass
class Options:
    """
    Class to hold all the options and hyperparameters for the training and evaluation.
    Some can be set from command line, others are set by default and can be changed from the script.
    """
    # Comman line parameters
    dataset_name: str = "norman_alt"
    project_name: str = "scgpt"
    split: str = "simulation"
    seed: int = 42
    lr: float = 1e-4
    batch_size: int = 4 # Recommended was 32 but not enought GPU
    eval_batch_size: int = 4
    epochs: int = 15
    num_runs: int = 1
    downsampling: bool = False

    # Other parameters that are more likely to be changed
    device: torch.device = field(default=None) # Will be set later unless want to set it here
    pretrained_model: bool = True # For now this only works with pretrained True

    # settings for optimizer
    schedule_interval: int = 1
    early_stop: int = 10

    # Model parameters
    load_param_prefixs: List[str] = field(default_factory=lambda: [
        "encoder",
        "value_encoder",
        "transformer_encoder",
    ])

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

    # Settings for model training
    MLM: bool = True  # whether to use masked language modeling, currently it is always on.
    CLS: bool = False  # celltype classification objective
    CCE: bool = False  # Contrastive cell embedding objective
    MVC: bool = False  # Masked value prediction for cell embedding
    ECS: bool = False  # Elastic cell similarity objective
    amp: bool = True

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
    Parses command-line arguments and overrides configuration values as needed.

    Configuration priority is as follows:
    Command-line arguments > JSON config file > default values.

    Supported arguments:
        --config (str): Path to a JSON configuration file. Primarily used to load settings from foundation models.
        --dataset_name (str): Name of the dataset to be used.
        --split (str): Dataset split strategy. Should match a function implemented in the `gears` library.
        --seed (int): Random seed for reproducibility.
        --learning_rate (float): Learning rate for the optimizer.
        --batch_size (int): Batch size for training. Recommended to use a higher value due to scGPT's GPU demands.
        --eval_batch_size (int): Batch size for evaluation.
        --epochs (int): Number of training epochs.
        --num_runs (int): Number of training iterations with different seeds. It is recommended to use 1 for now, as Transformer models are GPU intensive and it is not recommended to run multiple training iterations at the same time.

    Returns:
        argparse.Namespace: Parsed arguments with resolved configuration.
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

    parser.add_argument('--seed', type=int,
                        default=json_config.get('seed', 42),
                        help='Random seed for reproduciibility.')

    parser.add_argument('--learning_rate', type=float,
                        default=json_config.get('learning_rate', 1e-4),
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

    parser.add_argument('--num_runs', type=int,
                    default=1,
                    help='Number of training iterations with differente seeds.')

    parser.add_argument('--downsampling',
                        action='store_true',
                        help='Downsample dataset to keep equal number of samples for each condition.')

    # Parse all arguments (command line overrides JSON/defaults)
    return parser.parse_args(remaining_args)



def load_foundation_model(opts,
                          pert_data: PertData,
                          model_path: str,
                          pretrained_model: bool,
                          logger):
    """
    Loads a pretrained foundation model from scGPT to initialize the perturbation model.

    This model serves as the starting point for fine-tuning on the perturbation task.

    Args:
        opts: Configuration object containing model and training hyperparameters.
        pert_data (PertData): Object containing input data and metadata.
        model_path (str): Path to the pretrained foundation model checkpoint.
        pretrained_model (bool): If True, loads weights from a pretrained model. (Currently always set to True.)
        logger: Logger instance used for recording progress and messages.

    Returns:
        model: The initialized (and optionally pretrained) Transformer model.
        loaded_model_configs (dict): Dictionary containing model configurations such as gene_ids, vocab, etc.
    """

    scgpt_model_file = os.path.join(model_path, "best_model.pt")
    # Define configs for model

    loaded_model_configs = model_config_loading(pretrained_model, model_path, pert_data, opts.special_tokens, logger)
    # Set model configurations from configuration file if available
    if loaded_model_configs["load_from_config_file"]:
        model_configs = loaded_model_configs["model_configs"]
        embsize = model_configs["embsize"]
        d_hid = model_configs["d_hid"]
        nlayers = model_configs["nlayers"]
        nhead = model_configs["nheads"]
        n_layers_cls = model_configs["n_layers_cls"]

    vocab = loaded_model_configs["vocab"]

    # Review (Some setting didn't work, make sure tp recheck)
    embsize = opts.embsize  # embedding dimension
    d_hid = opts.d_hid  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = opts.nlayers  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = opts.nhead  # number of heads in nn.MultiheadAttention
    n_layers_cls = opts.n_layers_cls
    dropout = opts.dropout  # dropout probability
    use_fast_transformer = opts.use_fast_transformer  # whether to use fast transformer

    # Create and train scGPT
    ntokens = len(vocab)  # size of vocabulary

    # Get model architecture based on config
    model = TransformerGenerator(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        nlayers_cls=n_layers_cls,
        n_cls=1,
        vocab=vocab,
        dropout=dropout,
        pad_token=opts.pad_token,
        pad_value=opts.pad_value,
        pert_pad_id=opts.pert_pad_id,
        use_fast_transformer=use_fast_transformer,
    )

    if pretrained_model:
        model = load_pretrained(model, torch.load(scgpt_model_file), logger=logger)
    else:
        raise ValueError("Pretrained model is not set to True. Currently this script only works with the pretrained foundation model.")

    return model, loaded_model_configs


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



def train_perturbation_model(
    opts, 
    loaded_model_configs, 
    model: TransformerGenerator, 
    device: torch.device, 
    pert_data: PertData, 
    logger: scg.logger, 
    save_dir: str
):
    """
    Trains a Transformer-based perturbation model using perturbation gene expression data.

    Args:
        opts: Namespace object containing training hyperparameters and flags.
        loaded_model_configs (dict): Dictionary with model configuration (gene_ids, n_genes, etc.).
        model (TransformerGenerator): The transformer model to train.
        device (torch.device): The device to train on (CPU or GPU).
        pert_data (PertData): Object containing training and validation data loaders.
        logger (scg.logger): Logger object for tracking training progress.
        save_dir (str): Directory where the best model will be saved.

    Returns:
        best_model (TransformerGenerator): The model instance with the best validation score.
    """

    epochs = opts.epochs

    # Loss functions
    criterion = masked_mse_loss                    # Custom loss for masked regression
    criterion_cls = nn.CrossEntropyLoss()          # Standard classification loss (may be used optionally)

    # Optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=opts.schedule_interval, gamma=0.9
    )

    # Mixed-precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=opts.amp)

    # Track best model by Pearson correlation on validation set
    best_val_loss = float("inf")
    best_val_corr = 0
    best_model = None
    patience = 0

    # Model-specific configuration
    gene_ids = loaded_model_configs["gene_ids"]
    genes = loaded_model_configs["genes"]
    n_genes = loaded_model_configs["n_genes"]

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train_loader = pert_data.dataloader["train_loader"]
        valid_loader = pert_data.dataloader["val_loader"]

        clear_track_cache()

        # Run training for one epoch
        train(
            model=model,
            train_loader=train_loader,
            device=device,
            n_genes=n_genes,
            include_zero_gene=opts.include_zero_gene,
            max_seq_len=opts.max_seq_len,
            scaler=scaler,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            CLS=opts.CLS,
            CCE=opts.CCE,
            MVC=opts.MVC,
            ECS=opts.ECS,
            log_interval=opts.log_interval,
            logger=logger,
            epoch=epoch,
            gene_ids=gene_ids,
            amp=opts.amp
        )

        logger.info(f"Epoch {epoch} training finished")   

        clear_track_cache()

        # Evaluate model on validation set
        val_metrics = validate_perturbation_model(
            opts, gene_ids, model, pert_data, valid_loader, device
        ) 
        logger.info(f"val_metrics at epoch {epoch}: ")
        logger.info(val_metrics)

        elapsed = time.time() - epoch_start_time
        logger.info(f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | ")

        clear_track_cache()

        # Save model if current validation Pearson correlation is the best so far
        val_score = val_metrics["pearson"]
        if val_score > best_val_corr:
            best_val_corr = val_score
            best_model = copy.deepcopy(model)
            logger.info(f"Best model with score {val_score:5.4f}")
            patience = 0

            print(f"Saving best model with score {val_score:5.4f} at epoch {epoch} at {save_dir}.")
            torch.save(best_model.state_dict(), os.path.join(save_dir, "best_model.pt"))
        else:
            # Apply early stopping if validation score hasn't improved
            patience += 1
            if patience >= opts.early_stop:
                logger.info(f"Early stop at epoch {epoch}")
                return best_model

        # Step the learning rate scheduler
        scheduler.step()

    return best_model



def main(args: argparse.Namespace) -> None:
    """
    Main function to execute the training and evaluation of the model.
    """

    opts = Options(
        dataset_name=args.dataset_name,
        project_name=args.project_name,
        split=args.split,
        seed=args.seed,
        lr=args.learning_rate,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        num_runs=args.num_runs,
        downsampling=args.downsampling
    )


    scgpt_savedir = os.path.join(MODEL_DIR_PATH, opts.project_name)
    print(f"Training scGPT model {opts.num_runs} times with different seeds.")
    for current_seed in range(opts.seed, opts.seed + opts.num_runs):
        print(f"Running training with seed {current_seed}")

        # Checking and creating some directories
        # Directory where retrained model is saved

        save_dir = os.path.join(scgpt_savedir, f"scgpt_{opts.dataset_name}_{opts.split}_seed_{current_seed}")
        os.makedirs(save_dir, exist_ok=True)
        print(f"saving to {save_dir}")

        logger = scg.logger
        scg.utils.add_file_handler(logger, os.path.join(save_dir, "run.log"))
        # log running date and current git commit
        logger.info(f"Running on {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Configurations: {opts}")

        # Random seed
        set_seed(current_seed)
        # Set device
        if opts.device is not None:
            device = opts.device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Foundation model usage (for now only True)
        pretrained_model = opts.pretrained_model

        # Load dataset
        # single or double training is checked in opts: 'simulation' or 'simulation_single'
        pertdata = load_dataset(opts, current_seed, DATA_DIR_PATH, SINGLE_TRAIN_ONLY)

        # Load model based on configuration or new configurations
        model, loaded_model_configs = load_foundation_model(opts, pertdata, FOUNDATION_MODEL_PATH, pretrained_model, logger)
        model.to(device)

        # Training model, validating and keeping best model from validation
        best_model = train_perturbation_model(opts, loaded_model_configs, model, device, pertdata, logger, save_dir)

        print(f"Saving best model at {save_dir}.")
        torch.save(best_model.state_dict(), os.path.join(save_dir, "best_model.pt"))


if __name__ == '__main__':
    args = parse_args()
    main(args)
