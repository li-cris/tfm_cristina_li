import tempfile

import numpy as np

from tadvae.utils import load_gene_pathway_mask


def test_load_gene_pathway_mask():
    # Create a mock file with sample data.
    data = """pathway_id\tensembl_gene_id
    pathway1\tgene1
    pathway1\tgene2
    pathway2\tgene2
    pathway2\tgene3
    pathway3\tgene1
    pathway3\tgene3
    """
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write(data)
        map_file_path = f.name

    # Load the gene-pathway mask.
    mask, gene_to_index, pathway_to_index = load_gene_pathway_mask(map_file_path)

    # Expected outputs.
    expected_gene_to_index = {"gene1": 0, "gene2": 1, "gene3": 2}
    expected_pathway_to_index = {"pathway1": 0, "pathway2": 1, "pathway3": 2}
    expected_mask = np.array(
        [
            [1, 0, 1],  # gene1
            [1, 1, 0],  # gene2
            [0, 1, 1],  # gene3
        ]
    )

    # Check the outputs.
    assert gene_to_index == expected_gene_to_index
    assert pathway_to_index == expected_pathway_to_index
    assert np.array_equal(mask, expected_mask)
