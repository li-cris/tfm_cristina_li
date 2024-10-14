"""Functionality for handling perturbation datasets with metabolic model integration."""

import os
import sys

import globals
import matplotlib.pyplot as plt
import networkx as nx
import scanpy as sc
from models import init_model
from networkx.drawing.nx_agraph import graphviz_layout

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from transmet.pert_dataset import PertDataset


class MetaPertDataset(PertDataset):
    """Class for handling a perturbation dataset with metabolic model integration.

    Inherits from PertDataset and adds functionality for integrating a metabolic model.

    Attributes:
        metabolic_model: The metabolic model associated with the dataset.
    """

    def __init__(
        self,
        name: str,
        variant: str,
        dir_path: str,
        model_name: str = None,
        metabolic_model=None,
        **kwargs,
    ) -> "MetaPertDataset":
        """Initialize the MetaPertDataset object.

        Args:
            name: The name of the dataset.
            variant: The variant of the dataset.
            dir_path: The path to the datasets directory.
            model_name: The name of the metabolic model to load
                (optional if metabolic_model is provided).
            metabolic_model: A pre-loaded metabolic model (optional).
            **kwargs: Additional arguments for initializing the metabolic model.

        Returns:
            A MetaPertDataset object.
        """
        super().__init__(name, variant, dir_path)

        if metabolic_model is not None:
            print(
                f"Using provided metabolic model for dataset '{name}' "
                f"with variant '{variant}'."
            )
            self.metabolic_model = metabolic_model
        else:
            print(
                f"Loading metabolic model '{model_name}' for dataset '{name}' "
                f"with variant '{variant}'."
            )
            self.metabolic_model = load_metabolic_model(model_name, **kwargs)
            print(f"Metabolic model '{model_name}' loaded successfully.")

    def most_variable_genes(self, n_top_genes=5000, flavor="seurat") -> None:
        """Identify the most variable genes in the dataset.

        Args:
            n_top_genes: The number of most variable genes to identify.
            flavor: The flavor of the highly variable genes calculation.
        """
        # Create a temporary log-transformed copy of the adata object
        temp_log_adata = sc.pp.log1p(self.adata, copy=True)

        # Calculate highly variable genes on the temporary log-transformed data
        sc.pp.highly_variable_genes(
            adata=temp_log_adata, n_top_genes=n_top_genes, flavor=flavor
        )

        # Transfer the highly variable genes information to the original adata object
        self.adata.var["highly_variable"] = temp_log_adata.var["highly_variable"]
        self.adata.var["means"] = temp_log_adata.var["means"]
        self.adata.var["dispersions"] = temp_log_adata.var["dispersions"]
        self.adata.var["dispersions_norm"] = temp_log_adata.var["dispersions_norm"]

        # Check if 'highly_variable_rank' exists before transferring
        if "highly_variable_rank" in temp_log_adata.var:
            self.adata.var["highly_variable_rank"] = temp_log_adata.var[
                "highly_variable_rank"
            ]


def load_metabolic_model(model_name: str, **kwargs) -> None:
    """Load the metabolic model associated with the dataset.

    Args:
        model_name: The name of the metabolic model to load.
        **kwargs: Additional arguments for initializing the metabolic model.
    """
    # Default arguments for the metabolic model
    default_args = {
        "species": "homo_sapiens",
        "media": "default-media",
        "isoform_summing": "remove-summing",
        "exchange_limit": globals.EXCHANGE_LIMIT,
    }

    # Update default arguments with any provided kwargs
    default_args.update(kwargs)

    print(f"Initializing metabolic model '{model_name}' with the following parameters:")
    for key, value in default_args.items():
        print(f"  {key}: {value}")

    metabolic_model = init_model(
        model=model_name,
        species=default_args["species"],
        exchange_limit=default_args["exchange_limit"],
        media=default_args["media"],
        isoform_summing=default_args["isoform_summing"],
    )

    print("Metabolic model initialized successfully.")

    # Convert gene symbols to ensembl ids
    print("Converting gene symbols to Ensembl IDs...")
    metabolic_model.convert_gene_symbols_to_ensembl_ids()
    print("Gene symbols converted to Ensembl IDs.")

    # Remove empty gene associations
    print("Removing empty gene associations...")
    metabolic_model.remove_empty_gene_associations()
    print("Empty gene associations removed.")

    print("Metabolic model loading complete.")

    return metabolic_model


def print_dataset_metabolic_info(
    meta_pert_data, top_n_subsystems=10, variance_type="dispersions"
) -> None:
    """Print information about the metabolic genes in the perturbation dataset."""
    print("Metabolic Model Information Report")
    print("=" * 40)

    # Get the metabolic genes of the dataset.
    metabolic_model_genes = meta_pert_data.metabolic_model.get_genes()

    # Get the perturbation genes from the index
    perturbation_genes = meta_pert_data.adata.var.index.tolist()

    # Get the metabolic genes that are in the perturbation dataset.
    metabolic_genes_in_perturbation = [
        gene for gene in perturbation_genes if gene in metabolic_model_genes
    ]

    # Print the number of metabolic genes in the perturbation dataset.
    print(
        f"Number of metabolic genes in the perturbation dataset: "
        f"{len(metabolic_genes_in_perturbation)}"
    )

    # Identify the most variable genes
    most_variable_genes = meta_pert_data.adata.var[
        meta_pert_data.adata.var["highly_variable"]
    ].index.tolist()

    # Get the metabolic genes that are among the most variable genes
    metabolic_genes_in_most_variable = [
        gene for gene in most_variable_genes if gene in metabolic_model_genes
    ]

    # Print the number of metabolic genes among the most variable genes
    print(
        f"Number of metabolic genes among the most variable genes: "
        f"{len(metabolic_genes_in_most_variable)}"
    )

    # Fill NaN values with zero to avoid numerical problems
    meta_pert_data.adata.var[variance_type] = meta_pert_data.adata.var[
        variance_type
    ].fillna(0)

    # Calculate the total variance of all genes
    total_variance_all_genes = meta_pert_data.adata.var[variance_type].sum()

    # Calculate the variance captured by highly variable genes
    total_variance_highly_variable = meta_pert_data.adata.var[variance_type][
        meta_pert_data.adata.var["highly_variable"]
    ].sum()

    # Calculate the percentage of variance captured by highly variable genes
    # among all genes
    variance_percentage_highly_variable = (
        total_variance_highly_variable / total_variance_all_genes
    ) * 100

    # Print the percentage of variance captured by highly variable genes among all genes
    print(
        f"Percentage of variance captured by highly variable genes among all genes: "
        f"{variance_percentage_highly_variable:.2f}%"
    )

    # Calculate the variance captured by metabolic genes among the highly variable genes
    metabolic_variance = meta_pert_data.adata.var[variance_type][
        meta_pert_data.adata.var.index.isin(metabolic_genes_in_most_variable)
    ].sum()

    # Calculate the percentage of variance captured by metabolic genes among
    # the highly variable genes
    variance_percentage_metabolic = (
        metabolic_variance / total_variance_highly_variable
    ) * 100

    # Print the percentage of variance captured by metabolic genes among the
    # highly variable genes
    print(
        f"Percentage of variance captured by metabolic genes among highly variable "
        f"genes: {variance_percentage_metabolic:.2f}%"
    )

    # Calculate subsystems with metabolic genes in perturbation data
    subsystems_with_metabolic_genes_in_perturbation = [
        subsystem
        for subsystem in meta_pert_data.metabolic_model.subsystems.values()
        if any(
            gene in metabolic_genes_in_perturbation
            for gene in subsystem.get_associated_genes()
        )
    ]

    # Calculate subsystems with metabolic genes in most variable genes
    subsystems_with_metabolic_genes_in_most_variable = [
        subsystem
        for subsystem in meta_pert_data.metabolic_model.subsystems.values()
        if any(
            gene in metabolic_genes_in_most_variable
            for gene in subsystem.get_associated_genes()
        )
    ]

    total_subsystems = len(meta_pert_data.metabolic_model.subsystems)

    # Print the number of subsystems with metabolic genes in perturbation data
    print(
        f"Number of subsystems with metabolic genes in the perturbation dataset: "
        f"{len(subsystems_with_metabolic_genes_in_perturbation)}/{total_subsystems}"
    )

    # Print the number of subsystems with metabolic genes in most variable genes
    print(
        f"Number of subsystems with metabolic genes in the most variable genes: "
        f"{len(subsystems_with_metabolic_genes_in_most_variable)}/{total_subsystems}"
    )

    # Calculate variance captured by each subsystem with metabolic genes
    # in most variable genes
    subsystem_variance = {}
    seen_genes = set()
    for subsystem in subsystems_with_metabolic_genes_in_most_variable:
        subsystem_genes = subsystem.get_associated_genes()
        unique_genes = [gene for gene in subsystem_genes if gene not in seen_genes]
        seen_genes.update(unique_genes)
        subsystem_variance[subsystem.name] = meta_pert_data.adata.var[variance_type][
            meta_pert_data.adata.var.index.isin(unique_genes)
        ].sum()

    # Sort subsystems by variance captured and get the top N
    top_subsystems = sorted(
        subsystem_variance.items(), key=lambda x: x[1], reverse=True
    )[:top_n_subsystems]

    # Calculate total variance captured by metabolic genes
    total_metabolic_variance = sum(subsystem_variance.values())

    # Print the top N subsystems by variance captured
    print(f"\nTop {top_n_subsystems} Subsystems by Variance Captured:")
    print("-" * 40)
    for subsystem_name, variance in top_subsystems:
        subsystem = meta_pert_data.metabolic_model.subsystems[subsystem_name]
        subsystem_genes = subsystem.get_associated_genes()
        num_genes_in_most_variable = sum(
            gene in metabolic_genes_in_most_variable for gene in subsystem_genes
        )
        total_genes = len(subsystem_genes)
        variance_percentage = (variance / total_metabolic_variance) * 100
        print(
            f"Subsystem: {subsystem_name}, Variance: {variance:.2f}, "
            f"Percentage of metabolic variance: {variance_percentage:.2f}%, "
            f"Genes in Most Variable: {num_genes_in_most_variable}/{total_genes}"
        )

    # Print the total percentage of variance captured by the top N subsystems
    top_n_total_variance = sum(variance for _, variance in top_subsystems)
    top_n_variance_percentage = (top_n_total_variance / total_metabolic_variance) * 100
    print(
        f"\nTotal percentage of metabolic variance captured by the top "
        f"{top_n_subsystems} subsystems: {top_n_variance_percentage:.2f}%"
    )
    print("=" * 40)


def plot_subsystem_graph(
    subsystem_name: str,
    meta_pert_data: MetaPertDataset,
    base_figsize: tuple = (10, 10),
    omit_single_direction_metabolites: bool = False,
    differentiate_highly_variable_genes: bool = False,
):
    """Plot a directed weighted graph of the given subsystem.

    Args:
        subsystem_name (str): The name of the subsystem.
        meta_pert_data (MetaPertDataset): The MetaPertDataset object.
        base_figsize (tuple): Base size of the figure.
        omit_single_direction_metabolites (bool): Whether to omit metabolites that are
            only inputs or only outputs.
        differentiate_highly_variable_genes (bool): Whether to differentiate
            highly variable genes.
    """
    # Create a directed graph
    g = nx.DiGraph()

    # Get the metabolic model from the MetaPertDataset object
    metabolic_model = meta_pert_data.metabolic_model

    # Get the subsystem
    subsystem = metabolic_model.subsystems.get(subsystem_name)
    if not subsystem:
        print(f"Subsystem '{subsystem_name}' not found in the model.")
        return

    # Check if highly variable genes information is available
    highly_variable_genes = set()
    if (
        differentiate_highly_variable_genes
        and "highly_variable" in meta_pert_data.adata.var
    ):
        highly_variable_genes = set(
            meta_pert_data.adata.var.index[meta_pert_data.adata.var["highly_variable"]]
        )

    # Function to ensure node names are valid strings
    def valid_node_name(name, original_object, node_type):
        if name is None or name == "":
            print(f"Invalid {node_type} node name: {original_object}")
            return False
        return True

    # Add nodes and edges to the graph
    for reaction in subsystem.reactions:
        if not valid_node_name(reaction.id, reaction, "reaction"):
            continue
        reaction_id = str(reaction.id)
        g.add_node(
            reaction_id, label=reaction_id, shape="diamond", color="#ffcccb"
        )  # Reactions as diamonds

        # Add genes associated with the reaction
        has_highly_variable_gene = False
        for gene in reaction.list_genes():
            if not valid_node_name(gene, gene, "gene"):
                continue
            gene_id = str(gene)
            if gene_id in highly_variable_genes:
                gene_color = "#ffd700"  # Yellow for highly variable genes
                edge_color = "#ffd700"  # Yellow edges
                has_highly_variable_gene = True
            else:
                gene_color = "#d3d3d3"  # Gray for non-highly variable genes
                edge_color = "#d3d3d3"  # Gray edges

            g.add_node(gene_id, label=gene_id, shape="rect", color=gene_color)
            g.add_edge(gene_id, reaction_id, weight=1, color=edge_color)

        # Color the reaction gray if it has no highly variable genes associated
        if differentiate_highly_variable_genes and not has_highly_variable_gene:
            g.nodes[reaction_id]["color"] = "#d3d3d3"

        # Add reactants and products
        for metabolite, weight in reaction.reactants.items():
            if not valid_node_name(metabolite, metabolite, "metabolite"):
                continue
            metabolite_id = str(metabolite)
            g.add_node(
                metabolite_id, label=metabolite_id, shape="circle", color="#add8e6"
            )  # Metabolites as circles
            edge_color = (
                "#d3d3d3" if g.nodes[reaction_id]["color"] == "#d3d3d3" else "#e41a1c"
            )
            g.add_edge(
                metabolite_id, reaction_id, weight=weight, color=edge_color
            )  # Red or gray edges

        for metabolite, weight in reaction.products.items():
            if not valid_node_name(metabolite, metabolite, "metabolite"):
                continue
            metabolite_id = str(metabolite)
            g.add_node(
                metabolite_id, label=metabolite_id, shape="circle", color="#add8e6"
            )  # Metabolites as circles
            edge_color = (
                "#d3d3d3" if g.nodes[reaction_id]["color"] == "#d3d3d3" else "#377eb8"
            )
            g.add_edge(
                reaction_id, metabolite_id, weight=weight, color=edge_color
            )  # Blue or gray edges

    # Identify and color single direction metabolites if the flag is set
    if omit_single_direction_metabolites:
        to_remove = set()
        for node in g.nodes:
            if g.nodes[node]["shape"] == "circle":  # Metabolites
                in_degree = g.in_degree(node)
                out_degree = g.out_degree(node)
                if in_degree == 0 or out_degree == 0:
                    to_remove.add(node)

        if to_remove:
            print(
                (
                    f"Omitting the following single direction metabolites: "
                    f"{', '.join(to_remove)}"
                )
            )
            g.remove_nodes_from(to_remove)

    # Color metabolites gray if they have no input edges from highly variable reactions
    # or output edges to highly variable reactions
    for node in g.nodes:
        if g.nodes[node]["shape"] == "circle":  # Metabolites
            has_highly_variable_input = any(
                g.nodes[predecessor]["color"] != "#d3d3d3"
                for predecessor in g.predecessors(node)
            )
            has_highly_variable_output = any(
                g.nodes[successor]["color"] != "#d3d3d3"
                for successor in g.successors(node)
            )
            if not has_highly_variable_input and not has_highly_variable_output:
                g.nodes[node]["color"] = "#d3d3d3"

    # Ensure all node names are strings and attributes are properly handled
    for node in g.nodes:
        if not isinstance(node, str):
            print(f"Node {node} is not a string. Converting to string.")
            g = nx.relabel_nodes(g, {node: str(node)})

    for node, data in g.nodes(data=True):
        for key, value in data.items():
            if value is None:
                print(
                    f"Node {node} has attribute {key} with value None. "
                    f"Setting to empty string."
                )
                data[key] = ""

    # Plot each connected component separately
    for component in nx.weakly_connected_components(g):
        subgraph = g.subgraph(component)
        pos = graphviz_layout(
            subgraph, prog="dot"
        )  # positions for all nodes using dot layout

        # Calculate figure size based on the number of nodes
        num_nodes = len(subgraph.nodes)
        figsize = (base_figsize[0] + num_nodes * 0.3, base_figsize[1] + num_nodes * 0.3)

        # Create a new figure with the calculated size
        plt.figure(figsize=figsize)

        # Draw nodes with different shapes and colors
        node_shapes = {
            "circle": "o",
            "rect": "s",
            "diamond": "d",
        }  # Use 's' for rectangles
        for shape in node_shapes:
            nx.draw_networkx_nodes(
                subgraph,
                pos,
                nodelist=[
                    n for n in subgraph.nodes if subgraph.nodes[n]["shape"] == shape
                ],
                node_shape=node_shapes[shape],
                node_color=[
                    subgraph.nodes[n]["color"]
                    for n in subgraph.nodes
                    if subgraph.nodes[n]["shape"] == shape
                ],
                label=[
                    subgraph.nodes[n]["label"]
                    for n in subgraph.nodes
                    if subgraph.nodes[n]["shape"] == shape
                ],
                node_size=1000,  # Smaller node size for better visibility
            )

        # Draw edges with weights and transparency
        edges = subgraph.edges(data=True)
        for u, v, d in edges:
            nx.draw_networkx_edges(
                subgraph,
                pos,
                edgelist=[(u, v)],
                arrowstyle="-|>",
                arrowsize=20,
                width=2,
                alpha=0.5,  # Set transparency for all edges
                edge_color=d["color"],
            )

        edge_labels = {(u, v): d["weight"] for u, v, d in edges}
        nx.draw_networkx_edge_labels(
            subgraph, pos, edge_labels=edge_labels, font_size=8
        )

        # Draw node labels with smaller font size
        nx.draw_networkx_labels(
            subgraph,
            pos,
            labels={n: subgraph.nodes[n]["label"] for n in subgraph.nodes},
            font_size=10,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
        )

        # Show the plot for the current component
        plt.title(f"Subsystem: {subsystem_name} - Component")
        plt.axis("off")
        plt.show()
