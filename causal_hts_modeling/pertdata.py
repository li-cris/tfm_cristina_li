"""Functionality for handling perturbation data."""

import logging
import os

import scanpy as sc

from .utils import download_file, extract_zip

log = logging.getLogger(__name__)


class PertData:
    """
    Class for loading and processing perturbation data.

    The following are the [Gene Expression Omnibus](https://www.ncbi.nlm.nih.gov/geo/)
    accession numbers used:
    - Dixit et al., 2016: [GSE90063](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90063)
    - Adamson et al., 2016: [GSE90546](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90546)
    - Norman et al., 2019: [GSE133344](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133344)

    The following are the DOIs of the corresponding publications:
    - Dixit et al., 2016: https://doi.org/10.1016/j.cell.2016.11.038
    - Adamson et al., 2016: https://doi.org/10.1016/j.cell.2016.11.048
    - Norman et al., 2019: https://doi.org/10.1126/science.aax4438

    In general, in a perturbation dataset, we find N cell lines. Usually, one cell
    line remains unperturbed, and the others are cultivated separately (with different
    perturbations, i.e., gene knockouts). The mRNA of usually a few thousand  cells of
    each cell line is sequenced (using a single-cell RNA sequencing protocol),
    generating the gene expression profiles. Also, labels are available (i.e., we know
    which genes were knocked out in each cell line).
    """

    def __init__(
        self, data_dir: str, dataset_name: str, fix_labels: bool = True
    ) -> None:
        """
        Initialize the PertData object.

        Args:
            data_dir: The directory to save the data.
            dataset_name: The name of the dataset (supported: "dixit", "adamson",
                "norman").
            fix_labels: Whether to fix the perturbation labels. Default is True.
        """
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.dataset_path = os.path.join(self.data_dir, dataset_name)
        self.dataset_data = None
        self.fix_labels = fix_labels
        self.dataset_filtered = False
        self.X = None
        self.y = None

        if not os.path.exists(path=self.data_dir):
            log.info(f"Creating data directory: {self.data_dir}")
            os.makedirs(name=self.data_dir)
        else:
            log.info(f"Data directory already exists: {self.data_dir}")

        self._load()

    def _load(self) -> None:
        """Load perturbation dataset."""
        if self.dataset_name == "dixit":
            url = "https://dataverse.harvard.edu/api/access/datafile/6154416"
            dataset_data_filename = "dixit/perturb_processed.h5ad"
        elif self.dataset_name == "adamson":
            url = "https://dataverse.harvard.edu/api/access/datafile/6154417"
            dataset_data_filename = "adamson/perturb_processed.h5ad"
        elif self.dataset_name == "norman":
            url = "https://dataverse.harvard.edu/api/access/datafile/6154020"
            dataset_data_filename = "norman/perturb_processed.h5ad"
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        # Download and extract the dataset
        log.info(f"Downloading and extracting dataset: {self.dataset_name}")
        zip_filename = f"{self.dataset_path}.zip"
        download_file(url=url, save_filename=zip_filename)
        extract_zip(zip_path=zip_filename, extract_dir=self.dataset_path)

        # Load the dataset
        log.info(f"Loading dataset: {self.dataset_name}")
        self.dataset_data = sc.read_h5ad(
            filename=os.path.join(self.dataset_path, dataset_data_filename)
        )
        self.X = self.dataset_data.X
        self.y = self.dataset_data.obs["condition"]

        # Fix the perturbations labels
        if self.fix_labels:
            log.info("Fixing perturbation labels")
            self._fix_perturbation_labels()

    def _fix_perturbation_labels(self) -> None:
        """
        Fix the perturbation labels.

        In the perturbation datasets, single-gene perturbations are expressed as:
        - ctrl+<gene1>
        - <gene1>+ctrl

        Double-gene perturbations are expressed as:
        - <gene1>+<gene2>

        However, in general, there could also be multi-gene perturbations, and they
        might be expressed as a string with additional superfluous "ctrl+" in the
        middle:
            - ctrl+<gene1>+ctrl+<gene2>+ctrl+<gene3>+ctrl

        Hence, we need to remove superfluous "ctrl+" and "+ctrl" matches, such that
        perturbations are expressed as:
        - <gene1> (single-gene perturbation)
        - <gene1>+<gene2> (double-gene perturbation)
        - <gene1>+<gene2>+...+<geneN> (multi-gene perturbation)

        Note: Control cells are not perturbed and are labeled as "ctrl". We do not
        modify these labels.
        """
        # Remove "ctrl+" and "+ctrl" matches
        self.y = self.y.str.replace(pat="ctrl+", repl="")
        self.y = self.y.str.replace(pat="+ctrl", repl="")

    def log_info(self) -> None:
        """Log information about the dataset."""
        log.info("Perturbation dataset information:")
        log.info(f"  Dataset name: {self.dataset_name}")
        log.info(f"  Dataset path: {self.dataset_path}")
        log.info(f"  Labels fixed: {self.fix_labels}")
        log.info(f"  Dataset filtered: {self.dataset_filtered}")
        log.info(f"  Perturbations vector shape: {self.y.shape}")
        log.info(f"  Gene expression matrix shape: {self.X.shape}")
        log.info(f"  Different perturbations: {len(self.y.unique())}")

    def _filter_perturbations(self, pattern: str, negate: bool = False) -> None:
        """
        Filter perturbations and gene expressions.

        When a perturbation vector entry matches the pattern, the perturbation vector
        entry and the corresponding row from the gene expression matrix are kept and
        the rest are removed. This is the default case. If negate is True, the entries
        that match the pattern are removed and the rest are kept.

        Args:
            pattern: The pattern to match.
            negate: Whether to negate the pattern.
        """
        # Set up the filter mask
        filter_mask = self.y.str.contains(pat=pattern)
        if negate:
            filter_mask = ~filter_mask
        true_count = filter_mask.sum()
        false_count = len(filter_mask) - true_count

        # Apply the filter
        self.X = self.X[filter_mask, :]
        self.y = self.y[filter_mask]
        self.dataset_filtered = True

        # Log the filter
        log.info(f"Filter applied: pattern='{pattern}, negate={negate}")
        log.info(f"{true_count} remaining, {false_count} removed")

    def filter_only_single_gene_perturbations_(self) -> None:
        """
        Filter out everything except for single-gene perturbations.

        This method modifies the perturbations and gene expressions in place. It also
        requires the perturbation labels to be fixed.

        Raises:
            AssertionError: If perturbation labels are not fixed.
        """
        assert self.fix_labels, "Perturbation labels must be fixed"

        # Filter out perturbations that are equal to "ctrl"
        self._filter_perturbations(pattern="^ctrl$", negate=True)

        # Filter out double-gene perturbations
        self._filter_perturbations(pattern="\+", negate=True)
