"""Fetch information for all protein-coding genes in the human genome from Ensembl."""

import argparse

import pandas as pd
import requests


def _fetch_human_genes_info() -> pd.DataFrame:
    """Fetch information for all genes in the human genome from Ensembl.

    Returns:
        A DataFrame containing information about all genes in the human genome.
    """
    url = "https://www.ensembl.org/biomart/martservice"

    query = """<?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE Query>
    <Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="1" datasetConfigVersion="0.6">
        <Dataset name="hsapiens_gene_ensembl" interface="default">
            <Attribute name="ensembl_gene_id"/>
            <Attribute name="external_gene_name"/>
            <Attribute name="chromosome_name"/>
            <Attribute name="start_position"/>
            <Attribute name="end_position"/>
            <Attribute name="strand"/>
            <Attribute name="gene_biotype"/>
        </Dataset>
    </Query>"""

    try:
        response = requests.get(url, params={"query": query})
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to fetch information from Ensembl: {e}")

    # Read response as a DataFrame.
    lines = response.text.strip().split("\n")
    data = [line.split("\t") for line in lines[1:]]  # Skip header.
    df = pd.DataFrame(
        data,
        columns=[
            "ensembl_gene_id",
            "external_gene_name",
            "chromosome_name",
            "start_position",
            "end_position",
            "strand",
            "gene_biotype",
        ],
    )

    return df


def main():
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output_file_path", help="Path to save the output file.", required=True
    )
    args = parser.parse_args()

    # Fetch information for all genes in the human genome.
    genes_df = _fetch_human_genes_info()

    # Filter for protein-coding genes.
    protein_coding_genes_df = genes_df[genes_df["gene_biotype"] == "protein_coding"]

    # Save information to a file.
    protein_coding_genes_df.to_csv(args.output_file_path, sep="\t", index=False)
    print(f"Saved Ensembl gene info to: {args.output_file_path}")


if __name__ == "__main__":
    main()
