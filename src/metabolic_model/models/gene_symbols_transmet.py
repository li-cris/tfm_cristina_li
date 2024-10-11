"""This module provides functions to convert between Ensembl IDs and gene names."""

import json
import os
from typing import Dict, List

import requests

CACHE_DIR = "resources/Genes/"
CACHE_FILE = os.path.join(CACHE_DIR, "gene_cache.json")


def load_cache() -> Dict[str, str]:
    """Load the gene cache from a file."""
    if not os.path.exists(CACHE_FILE):
        return {}
    with open(CACHE_FILE, "r") as f:
        return json.load(f)


def save_cache(cache: Dict[str, str]) -> None:
    """Save the gene cache to a file."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)


def batch_gene_names_to_ensembl_ids(
    gene_names: List[str], batch_size: int = 1000, cache: Dict[str, str] = None
) -> Dict[str, str]:
    """Get the Ensembl IDs for a list of gene names using the Ensembl REST API.

    Args:
        gene_names: A list of gene names.
        batch_size: The maximum number of gene names to include in each request.
        cache: A dictionary containing cached gene information.

    Returns:
        A dictionary mapping gene names to Ensembl IDs.

    Raises:
        requests.exceptions.Request: If the request to the Ensembl REST API fails.
    """
    server = "https://rest.ensembl.org"
    ext = "/lookup/symbol/homo_sapiens"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    gene_name_to_ensembl = {}

    # Use the cache to avoid redundant requests
    if cache is None:
        cache = {}

    for i in range(0, len(gene_names), batch_size):
        batch = gene_names[i : i + batch_size]
        batch_to_request = [gene.upper() for gene in batch if gene.upper() not in cache]

        if not batch_to_request:
            print(f"All genes in batch {i // batch_size + 1} are found in cache.")
            continue

        try:
            print(
                f"Requesting Ensembl IDs for batch {i // batch_size + 1} from the internet."
            )
            # Manually construct the JSON string for the list of gene names
            data = json.dumps({"symbols": batch_to_request})

            # Send a POST request to the Ensembl REST API with the list of gene names.
            response = requests.post(server + ext, headers=headers, data=data)
            response.raise_for_status()

            # Parse the JSON response to get the mapping of gene names to Ensembl IDs.
            data = response.json()
            new_gene_info = {
                gene: details["id"] for gene, details in data.items() if "id" in details
            }
            gene_name_to_ensembl.update(new_gene_info)

            # Update the cache with the new gene information
            cache.update(new_gene_info)

            # Log any genes that were not found in the response
            missing_genes = [
                gene for gene in batch_to_request if gene not in new_gene_info
            ]
            if missing_genes:
                print(
                    f"Warning: The following genes were not found in the response: {missing_genes}"
                )

        except requests.exceptions.RequestException as e:
            raise ValueError(
                f"Failed to get Ensembl IDs from the Ensembl REST API: {e}"
            )

    # Include the cached information in the result
    gene_name_to_ensembl.update(
        {gene: cache[gene.upper()] for gene in gene_names if gene.upper() in cache}
    )

    return gene_name_to_ensembl


def get_ensembl_ids_with_cache(
    gene_names: List[str], batch_size: int = 1000
) -> Dict[str, str]:
    """Get the Ensembl IDs for a list of gene names, using the cache if available.

    Args:
        gene_names: A list of gene names.
        batch_size: The maximum number of gene names to include in each request.

    Returns:
        A dictionary mapping gene names to Ensembl IDs.
    """
    # Load the cache
    cache = load_cache()

    # Check if all genes are in the cache
    all_in_cache = all(gene.upper() in cache for gene in gene_names)

    if all_in_cache:
        print("All requested genes are found in the cache.")
        return {gene: cache[gene.upper()] for gene in gene_names}

    print(
        (
            "Some genes are not found in the cache. "
            "Requesting missing genes from the internet."
        )
    )

    # Print the genes that are not in the cache
    missing_genes = [gene for gene in gene_names if gene.upper() not in cache]
    print(f"Missing genes length: {len(missing_genes)}")
    # Get Ensembl IDs for the missing genes
    gene_name_to_ensembl = batch_gene_names_to_ensembl_ids(
        missing_genes, batch_size, cache
    )

    # Save the updated cache
    save_cache(cache)

    # Include the cached information in the result
    gene_name_to_ensembl.update(
        {gene: cache[gene.upper()] for gene in gene_names if gene.upper() in cache}
    )

    return gene_name_to_ensembl


def convert_gene_symbols_to_ensembl_ids(model):
    """Convert all gene symbols or names in the model to Ensembl IDs.

    Args:
        model: An instance of MetabolicModelTransmet.
    """
    # Collect all gene names in the model
    all_genes = set()
    for reaction in model.reactions.values():
        all_genes |= set(reaction.list_genes())

    # Convert gene names to Ensembl IDs using the cache
    gene_name_to_ensembl = get_ensembl_ids_with_cache(list(all_genes))

    # Update the gene associations in the model
    def update_association(assoc):
        if assoc.type == "gene":
            gene = assoc.gene
            if gene.name in gene_name_to_ensembl:
                gene.id = gene_name_to_ensembl[gene.name]
                gene.name = gene.id
        else:
            for child in assoc.children:
                update_association(child)

    for reaction in model.reactions.values():
        if reaction.gene_associations is not None:
            update_association(reaction.gene_associations)

    # Verify that all genes are cached
    cached_genes = load_cache()
    missing_genes = [gene for gene in all_genes if gene.upper() not in cached_genes]
    if missing_genes:
        print(f"Warning: The following genes were not cached: {missing_genes}")
    else:
        print("All genes have been successfully cached.")
