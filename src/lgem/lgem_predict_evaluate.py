import argparse
import torch
import numpy as np
import pandas as pd
import os
import scanpy as sc
import json

from typing import Dict

from lgem.models import (
    LinearGeneExpressionModelLearned,
    LinearGeneExpressionModelOptimized,
)
from lgem.utils import predict_evaluate_lgem_double, set_seeds, evaluate_double_metrics
from data_utils.single_norman_utils import separate_data
from lgem.lgem_run import parse_args


def load_previous_config(filepath: str) -> Dict:
    """
    Load configurations from training.
    """
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)

    return {}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Parse command line arguments.")
    # Directory where trained model and dataloaders are saved
    parser.add_argument(
        "-s",
        "--savedir",
        type=str,
        default="./models/",
        help="Main directory to save the trained models."
    )
    parser.add_argument(
        "--eval_dir",
        type=str,
        default="./results/",
        help="Main directory to save prediction results and evaluation metrics."
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/",
        help="Directory where the dataset is stored. Best to use global path here."
    )

    parser.add_argument(
        "--name",
        type=str,
        default="lgem",
        help="Name to give the project folder in main directories."
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Norman_2019raw",
        help="Name of the dataset to use for training and evaluation. Needs to be .h5ad file."
    )

    parser.add_argument(
        "--device", type=str, default=None, help="Chosen device in which run model."
    ) 

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed."
    )

    parser.add_argument(
        "--num_runs", type=int, default=1, help="Number of runs."
    )

    parser.add_argument(
        "--epochs", type=int, default=250, help="Number of epochs."
    )

    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size."
    )

    parser.add_argument(
        "--prediction_type", type=str, default="double", help="Kind of prediction to carry out in evaluation. Supported: 'double' or other."
    )

    parser.add_argument(
        "--top_deg", type=int, default=20, help="Number of Differentially Expressed Genes to evaluate."
    )

    parser.add_argument(
        "--pool_size", type=int, default=200, help="Number of control cells to randomly sample for evaluation."
    )

    return parser.parse_args()

def main(args):
    # Load previous configuration first
    config_path = os.path.join(args.savedir, "config.json")
    prev_args = load_previous_config(config_path)

    # Keep configurations from previous run
    seed = prev_args.get('seed', args.seed)
    num_runs=prev_args.get('num_runs', args.num_runs)
    device = prev_args.get('device', args.device)
    n_epochs=prev_args.get('epochs', args.epochs)
    batch_size=prev_args.get('batch_size', args.batch_size)
    top_deg=prev_args.get('top_deg', args.top_deg)
    pool_size=prev_args.get('pool_size', args.pool_size)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seeds(seed)

    # Loading data for control samples
    dataset_name=prev_args.get('dataset_name', args.dataset_name)
    data_dir=prev_args.get('data_dir', args.data_dir)

    global_data_path=os.path.join(data_dir, dataset_name + ".h5ad")
    print(f"Loading dataset from {global_data_path}.")
    pertdata = sc.read(global_data_path)
    _, pertdata_double, pertdata_ctrl = separate_data(adata = pertdata, dataset_name = dataset_name)


    for current_run in range(num_runs):
        # Trained model path
        current_seed = seed + current_run
        model_name = f"lgem_{dataset_name}_seed_{current_seed}_epoch_{n_epochs}_batch_{batch_size}"
        savedir = os.path.join(args.savedir, model_name)

        # Load dataset (dataloader)
        test_dataloader = torch.load(os.path.join(savedir, "test_dataloader.pt"))
        perturbation_list = torch.load(os.path.join(savedir, "perts.pt"))
        perturbation_list = perturbation_list["perts"]

        # Load Embeddings
        G = torch.load(os.path.join(savedir, "G.pt"))
        P = torch.load(os.path.join(savedir, "P.pt"))
        Y = torch.load(os.path.join(savedir, "Y.pt"))
        b = torch.load(os.path.join(savedir, "b.pt"))

        # Optimized model
        model_optimized = LinearGeneExpressionModelOptimized(Y.T, G, P, b)
        model_optimized.load_state_dict(torch.load(os.path.join(savedir, "optimized_best_model.pt")))
        double_perts_list_op, double_predictions_op, ground_truth, mse_pred_op = predict_evaluate_lgem_double(model_optimized, device, test_dataloader, perturbation_list)

        # Learned model
        model_learned = LinearGeneExpressionModelLearned(G, b)
        model_learned.load_state_dict(torch.load(os.path.join(savedir, "learned_best_model.pt")))
        _, double_predictions_learn, _, mse_pred_learn= predict_evaluate_lgem_double(model_learned, device, test_dataloader, perturbation_list)    

        double_predictions_op = np.array(double_predictions_op)
        double_predictions_learn = np.array(double_predictions_learn)

        # Metrics for optmized model
        print("Evaluating metrics for optimized model.")
        evaluate_double_metrics(double_adata=pertdata_double, ctrl_adata=pertdata_ctrl,
                                predictions=double_predictions_op,
                                model_name=model_name, results_savedir=args.eval_dir,
                                double_perts=double_perts_list_op,
                                pool_size=pool_size, seed=current_seed, top_deg=top_deg,
                                model_type='op')

        # Metrics for learned model
        print("Evaluating metrics for learned model.")
        evaluate_double_metrics(double_adata=pertdata_double, ctrl_adata=pertdata_ctrl,
                                predictions=double_predictions_learn,
                                model_name=model_name, results_savedir=args.eval_dir,
                                double_perts=double_perts_list_op,
                                pool_size=pool_size, seed=current_seed, top_deg=top_deg,
                                model_type='learn')

        # Saving predictions
        double_pred_op = pd.DataFrame(double_predictions_op, columns=pertdata_ctrl.var_names)
        double_pred_op.insert(0, 'double', double_perts_list_op)
        double_pred_learn = pd.DataFrame(double_predictions_learn, columns=pertdata_ctrl.var_names)
        double_pred_learn.insert(0, 'double', double_perts_list_op)

        double_pred_op.to_csv(os.path.join(args.eval_dir, f"{model_name}_double_predictions_op.csv"), index=False)
        double_pred_learn.to_csv(os.path.join(args.eval_dir, f"{model_name}_double_predictions_learn.csv"), index=False)
        print("Prediction GEPs saved.")

        # # Randomly chosen control cells for baseline
        # rand_idx = np.random.randint(low=0, high=pertdata_ctrl.X.shape[0], size=len(double_perts_list_op))
        # baseline_control = pertdata_ctrl.X[rand_idx, :].toarray()

        # # Turning profiles into arrays
        # double_predictions_op = np.asarray(double_predictions_op)
        # double_predictions_learn = np.asarray(double_predictions_learn)
        # gt = np.asarray(ground_truth)
        # baseline_control = np.asarray(baseline_control) # Redundant I tink but it broke


        # DGEP True - Pred
    
        # DGEP True - Control

        # # Calculate true - pred MSE
        # mse_control_op = np.mean((gt - baseline_control) ** 2, axis = 1)
        # mse_control_learn = np.mean((gt - baseline_control) ** 2, axis = 1)
        # print("Finalised MSE calculations.")

        # # Calculate Pearson correlation
        # gt_deg = gt - baseline_control
        # deg_idx = np.argsort(abs(gt_deg), axis=1)[:, -top_deg:]

        # # Select values along the top DEG indices for each sample
        # pred_op_selected = np.take_along_axis(double_predictions_op - baseline_control, deg_idx, axis=1)
        # pred_learn_selected = np.take_along_axis(double_predictions_learn - baseline_control, deg_idx, axis=1)
        # gt_selected = np.take_along_axis(gt_deg, deg_idx, axis=1)

        # pearson_op = np.array([pearsonr(pred_op_selected[i], gt_selected[i])
        #                     for i in range(pred_op_selected.shape[0])
        #                     ])
        # pearson_learn = np.array([pearsonr(pred_learn_selected[i], gt_selected[i])
        #                     for i in range(pred_learn_selected.shape[0])
        #                ])

        # # Save metrics to result dir
        # result_df = pd.DataFrame({"double": double_perts_list_op,
        #                         "mse_true_vs_ctrl_op": mse_control_op,
        #                         "mse_true_vs_ctrl_learn": mse_control_learn,
        #                         "mse_true_vs_pred_op": mse_pred_op,
        #                         "mse_true_vs_pred_learn": mse_pred_learn,
        #                         f"PearsonTop{top_deg}_true_vs_pred_op": pearson_op[:, 0],
        #                         "Pearson_pval_true_vs_pred_op": pearson_op[:, 1],
        #                         f"PearsonTop{top_deg}_true_vs_pred_learn": pearson_learn[:, 0],
        #                         "Pearson_pval_true_vs_pred_learn": pearson_learn[:, 1]})




if __name__ == "__main__":
    args = parse_args()

    # Build the save directory path
    args.savedir = os.path.join(args.savedir, args.name)
    os.makedirs(args.savedir, exist_ok=True)

    args.eval_dir = os.path.join(args.eval_dir, args.name)
    os.makedirs(args.eval_dir, exist_ok=True)

    main(args)