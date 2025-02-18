"""Generate a GO pathway-genes map from MSigDB.

The input should be the file c5.go.bp.v2024.1.Hs.json.txt, downloaded from MSigDB.
This file is from collection C5, first subcollection, Biological Process (BP) component.

Links:
- MSigDB: https://www.gsea-msigdb.org/gsea/msigdb/index.jsp
- Collection C5: https://www.gsea-msigdb.org/gsea/msigdb/human/collections.jsp#C5
- BP component (JSON bundle): https://www.gsea-msigdb.org/gsea/msigdb/download_file.jsp?filePath=/msigdb/release/2024.1.Hs/c5.go.bp.v2024.1.Hs.json
"""

import argparse

import pandas as pd


def main():
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_file_path", help="Path to the input file.", required=True
    )
    parser.add_argument(
        "-e",
        "--ensembl_gene_info_file_path",
        help="Path to the Ensembl gene info file.",
        required=True,
    )
    parser.add_argument(
        "-o", "--output_file_path", help="Path to the output file.", required=True
    )
    args = parser.parse_args()

    # Load the input file and extract the pathway-genes map.
    df = pd.read_json(args.input_file_path)
    df = df.transpose()
    pathway_genes_map = df[["exactSource", "geneSymbols"]].explode("geneSymbols")

    # Load the Ensembl gene info file.
    ensembl_gene_info = pd.read_csv(args.ensembl_gene_info_file_path, sep="\t")

    # Merge the pathway_genes_map with the ensembl_gene_info to replace gene names with
    # Ensembl gene IDs.
    pathway_genes_map = pathway_genes_map.rename(
        columns={"exactSource": "pathway_id", "geneSymbols": "gene_name"}
    )
    merged_map = pathway_genes_map.merge(
        ensembl_gene_info,
        left_on="gene_name",
        right_on="external_gene_name",
        how="left",
    )
    merged_map = merged_map[["pathway_id", "ensembl_gene_id"]]

    # Discard rows with no matching Ensembl gene ID.
    merged_map = merged_map.dropna(subset=["ensembl_gene_id"])

    # Save the merged map to a file.
    merged_map.to_csv(args.output_file_path, sep="\t", index=False)


if __name__ == "__main__":
    main()
