from .data import load_data


def main():
    adata, gene_pathway_mask, gene_to_index, pathway_to_index = load_data(
        "NormanWeissman2019_filtered"
    )

    # TODO: Min-max normalization needed for expression values?


if __name__ == "__main__":
    main()
