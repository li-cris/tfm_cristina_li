import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import argparse

import torch

from sena2.utils import Norman2019DataLoader  # type: ignore


def generate_activation_scores(model_file_path: str, batch_size: int = 32) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_handler = Norman2019DataLoader(batch_size=batch_size)
    model = torch.load(model_file_path, weights_only=False, map_location=device)

    # load data and reset index
    adata = data_handler.adata
    adata.obs = adata.obs.reset_index(drop=True)
    print(adata.obs)
    ptb_targets = data_handler.ptb_targets
    gos = data_handler.gos

    """build pert idx dict"""
    data_handler.gene_var = "guide_ids"
    idx_dict = {}
    for knockout in ["ctrl"] + ptb_targets:
        if knockout == "ctrl":
            idx_dict[knockout] = (
                adata.obs[adata.obs[data_handler.gene_var] == ""]
            ).index.values
        else:
            idx_dict[knockout] = (
                adata.obs[adata.obs[data_handler.gene_var] == knockout]
            ).index.values

    # ##
    # n_pertb = len(ptb_targets)
    # pert_dict = {}
    # info_dict = defaultdict(lambda: defaultdict(list))
    # results_dict = {}

    # """compute"""
    # with torch.no_grad():
    #     for gene in tqdm(idx_dict, desc="generating activity score for perturbations"):
    #         idx = idx_dict[gene]
    #         mat = torch.from_numpy(adata.X[idx, :].todense()).to(device).double()

    #         """first layer"""

    #         na_score_fc1 = model.fc1(mat)
    #         info_dict["fc1"][gene].append(na_score_fc1.detach().cpu().numpy())

    #         """mean + var"""

    #         na_score_fc_mean = model.fc_mean(na_score_fc1)
    #         info_dict["fc_mean"][gene].append(na_score_fc_mean.detach().cpu().numpy())

    #         na_score_fc_var = torch.nn.Softplus()(model.fc_var(na_score_fc1))
    #         info_dict["fc_var"][gene].append(na_score_fc_var.detach().cpu().numpy())

    #         """reparametrization trick"""

    #         na_score_mu, na_score_var = model.encode(mat)
    #         na_score_z = model.reparametrize(na_score_mu, na_score_var)
    #         info_dict["z"][gene].append(na_score_z.detach().cpu().numpy())

    #         """causal graph"""

    #         if gene != "ctrl":
    #             # define ptb idx
    #             ptb_idx = np.where(np.array(ptb_targets) == gene)[0][0]

    #             # generate one-hot-encoder
    #             c = torch.zeros(size=(1, n_pertb))
    #             c[:, ptb_idx] = 1
    #             c = c.to(device).double()

    #             # decode an interventional sample from an observational sample
    #             bc, csz = model.c_encode(c, temp=1)
    #             bc2, csz2 = bc, csz
    #             info_dict["bc_temp1"][gene].append(bc.detach().cpu().numpy())

    #             # decode an interventional sample from an observational sample
    #             bc, csz = model.c_encode(c, temp=100)
    #             bc2, csz2 = bc, csz
    #             info_dict["bc_temp100"][gene].append(bc.detach().cpu().numpy())

    #             # decode an interventional sample from an observational sample
    #             bc, csz = model.c_encode(c, temp=1000)
    #             bc2, csz2 = bc, csz
    #             info_dict["bc_temp1000"][gene].append(bc.detach().cpu().numpy())

    #             # compute assignation
    #             if ptb_idx not in pert_dict:
    #                 pert_dict[ptb_idx] = bc[0].argmax().__int__()

    #             # interventional U
    #             na_score_u = model.dag(na_score_z, bc, csz, bc2, csz2, num_interv=1)
    #             info_dict["u"][gene].append(na_score_u.detach().cpu().numpy())

    #         else:
    #             # observational U
    #             na_score_u = model.dag(na_score_z, 0, 0, 0, 0, num_interv=0)
    #             info_dict["u"][gene].append(na_score_u.detach().cpu().numpy())

    # """build dataframes within each category"""
    # for layer in info_dict:
    #     temp_df = []
    #     for gene in info_dict[layer]:
    #         info_dict[layer][gene] = pd.DataFrame(np.vstack(info_dict[layer][gene]))
    #         info_dict[layer][gene].index = [gene] * info_dict[layer][gene].shape[0]
    #         temp_df.append(info_dict[layer][gene])

    #     # substitute
    #     results_dict[layer] = pd.concat(temp_df)
    #     if layer == "fc1":
    #         results_dict[layer].columns = gos

    # # add pertb_dict
    # results_dict["pert_map"] = pd.DataFrame(pert_dict, index=[0]).T
    # results_dict["pert_map"].columns = ["c_enc_mapping"]
    # results_dict["causal_graph"] = model.G.detach().cpu().numpy()

    # """add weights layers (delta) for """
    # results_dict["mean_delta_matrix"] = pd.DataFrame(
    #     model.fc_mean.weight.detach().cpu().numpy().T, index=gos
    # )
    # results_dict["std_delta_matrix"] = pd.DataFrame(
    #     model.fc_var.weight.detach().cpu().numpy().T, index=gos
    # )

    # """save info"""
    # with open(os.path.join("activation_scores.pickle"), "wb") as handle:
    #     pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate activation scores.")
    parser.add_argument(
        "--model_file_path", type=str, required=True, help="Path to the model file."
    )
    args = parser.parse_args()

    generate_activation_scores(args.model_file_path)
