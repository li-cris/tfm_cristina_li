from typing import List, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
import torch.nn as nn

def separate_data(adata = None, dataset_name = "norman"):
    """Get the single perturbation dataset, double perturbation dataset and control dataset from the given AnnData object as well as list of single perturbations."""
    if dataset_name == "norman":
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
    """Get adata with perts found in genes (features) and return list of perts and genes."""
    if dataset_name == "norman":
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


def compute_embeddings_double(
    Y: torch.Tensor,  # noqa: N803
    perts: List[str], # all perturbations, single and double
    genes: List[str],
    d_embed: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute gene and perturbation embeddings.

    Args:
        Y: Data matrix with shape (n_genes, n_perturbations).
        perts: List of perturbations.
        genes: List of genes.
        d_embed: Embedding dimension.

    Returns:
        G: Gene embedding matrix with shape (n_genes, d_embed).
        P: Perturbation embedding matrix with shape (n_perturbations, d_embed).
        b: Bias vector with shape (n_genes).
    """
    # Perform a PCA on Y to obtain the top d_embed principal components, which will
    # serve as the gene embeddings G.
    pca = PCA(n_components=d_embed)
    G = pca.fit_transform(Y)  # noqa: N806

    gene_to_idx = {gene: i for i, gene in enumerate(genes)}
    gene_to_emb = {gene: G[i] for gene, i in gene_to_idx.items()}

    P = []
    missing = []
    for pert in perts:
        genes_in_pert = pert.split("+")
        try:
            emb_list = [gene_to_emb[g] for g in genes_in_pert]
            pert_emb = np.mean(emb_list, axis=0)  # average embeddings
            P.append(pert_emb)
        except KeyError as e:
            missing.append(pert)
            P.append(np.zeros(d_embed))  # fallback if gene not found
    
    P = np.array(P)
    if missing:
        print(f"{len(missing)}/{len(perts)} missing embeddings.")
        print(f"Missing embeddings for perturbations: {missing}")

    # Compute b as the average expression of each gene across all perturbations.
    b = Y.mean(axis=1)

    return torch.from_numpy(G).float(), torch.from_numpy(P).float(), b


# evaluate model
def predict_lgem_singles(model, dataloader, device):
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_P, _ in dataloader:  # noqa: N806
            batch_P = batch_P.to(device)  # noqa: N806
            Y_predicted = model(batch_P)  # noqa: N806
            predictions.append(Y_predicted.cpu().numpy())

    single_predictions = np.concatenate(predictions, axis=1)
    return single_predictions

def predict_evaluate_lgem_double(model, device, dataloader, perts_list):
    """Predicts the double prediction output of the model based on embedding of double perturbations."""
    model.to(device)
    model.eval()
    double_perts_list = [pert for pert in perts_list if "+" in pert]
    predictions = []
    ground_truth = [] 
    mse_loss_list = []
    mse_loss_fn = nn.MSELoss(reduction = "none")
    with torch.no_grad():
        print("Predicting and calculating loss for double perturbations.")
        for batch_P, batch_Y in dataloader:  # noqa: N806
            batch_P, batch_Y = batch_P.to(device), batch_Y.to(device)  # noqa: N806
            Y_predicted = model(batch_P)  # noqa: N806
            mse_loss = mse_loss_fn(Y_predicted.T, batch_Y)
            predictions.append(Y_predicted.T.cpu().numpy())
            ground_truth.append(batch_Y.cpu().numpy())

            mse_loss_list.extend(mse_loss.cpu().numpy())


    double_predictions = np.concatenate(predictions, axis=0)
    ground_truth = np.concatenate(ground_truth, axis=0)

    return double_perts_list, double_predictions, ground_truth, mse_loss_list
