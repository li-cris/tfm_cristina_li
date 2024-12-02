"""Utility functions for the transmet package."""

import requests


def ensembl_id_to_gene_name(ensembl_id: str) -> str:
    """Get the gene name for an Ensembl ID using the Ensembl REST API.

    To map Ensembl stable IDs (such as gene, transcript, or protein IDs) to gene names
    (i.e., HGNC symbols from the
    [HUGO Gene Nomenclature Committee](https://www.genenames.org)), we use the Ensembl
    REST API.

    Args:
        ensembl_id: The Ensembl ID.

    Returns:
        The gene name.

    Raises:
        requests.exceptions.Request: If the request to the Ensembl REST API fails.
    """
    # The URL for the Ensembl REST API.
    url = (
        f"https://rest.ensembl.org/lookup/id/{ensembl_id}?content-type=application/json"
    )

    try:
        # Send a GET request to the Ensembl REST API.
        response = requests.get(url=url)
        response.raise_for_status()

        # Return the gene name from the JSON response.
        data = response.json()
        return data.get("display_name", "No gene name found")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to get gene name from the Ensembl REST API: {e}")
