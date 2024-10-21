"""Module for visualizing metabolic model subsystems as directed weighted graphs."""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
from meta_pert_dataset import MetaPertDataset

# Define constants for shapes and colors
REACTION_SHAPE = "diamond"
REACTION_COLOR = "#ffcccb"  # Light pink color
REACTION_GRAPH_COLOR = "#800080"  # Purple color
PRODUCT_EDGE_COLOR = "gray"
REACTANT_EDGE_COLOR = PRODUCT_EDGE_COLOR

METABOLITE_SHAPE = "circle"
METABOLITE_COLOR = "#6ACCFD"  # Light blue

GENE_SHAPE = "rect"
GENE_COLOR = "#F4A51D"  # Orange color
HIGHLY_VARIABLE_GENE_COLOR = "#F44B81"  # Pink color
PERTURBED_GENE_EDGE_COLOR = "#4BF44F"  # Green color

GRADIENT_MAP = plt.cm.viridis


def build_basic_subsystem_graph(
    subsystem_name: str, meta_pert_data: MetaPertDataset
) -> nx.DiGraph:
    """Build a directed weighted graph of the given subsystem."""
    g = nx.DiGraph()
    metabolic_model = meta_pert_data.metabolic_model
    subsystem = metabolic_model.subsystems.get(subsystem_name)
    if not subsystem:
        print(f"Subsystem '{subsystem_name}' not found in the model.")
        return g

    def valid_node_name(name, original_object, node_type):
        if name is None or name == "":
            print(f"Invalid {node_type} node name: {original_object}")
            return False
        return True

    reaction_nodes = []
    for reaction in subsystem.reactions:
        if not valid_node_name(reaction.id, reaction, "reaction"):
            continue
        reaction_id = str(reaction.id)
        g.add_node(
            reaction_id, label=reaction_id, shape=REACTION_SHAPE, color=REACTION_COLOR
        )
        reaction_nodes.append(reaction)

        for gene in reaction.list_genes():
            if not valid_node_name(gene, gene, "gene"):
                continue
            gene_id = str(gene)
            g.add_node(gene_id, label=gene_id, shape=GENE_SHAPE, color=GENE_COLOR)
            g.add_edge(gene_id, reaction_id, weight=1, color=GENE_COLOR, alpha=0.5)

        for metabolite, weight in reaction.reactants.items():
            if not valid_node_name(metabolite, metabolite, "metabolite"):
                continue
            metabolite_id = str(metabolite)
            g.add_node(
                metabolite_id,
                label=metabolite_id,
                shape=METABOLITE_SHAPE,
                color=METABOLITE_COLOR,
            )
            g.add_edge(
                metabolite_id,
                reaction_id,
                weight=weight,
                color=REACTANT_EDGE_COLOR,
                alpha=0.5,
            )

        for metabolite, weight in reaction.products.items():
            if not valid_node_name(metabolite, metabolite, "metabolite"):
                continue
            metabolite_id = str(metabolite)
            g.add_node(
                metabolite_id,
                label=metabolite_id,
                shape=METABOLITE_SHAPE,
                color=METABOLITE_COLOR,
            )
            g.add_edge(
                reaction_id,
                metabolite_id,
                weight=weight,
                color=PRODUCT_EDGE_COLOR,
                alpha=0.5,
            )

    single_direction_metabolites = set()
    for node in g.nodes:
        if g.nodes[node]["shape"] == METABOLITE_SHAPE:
            in_degree = g.in_degree(node)
            out_degree = g.out_degree(node)
            if in_degree == 0 or out_degree == 0:
                single_direction_metabolites.add(node)

    print(f"Single-direction metabolites: {single_direction_metabolites}")

    for r1 in reaction_nodes:
        for r2 in reaction_nodes:
            if r1 != r2 and set(r1.products).intersection(set(r2.reactants)):
                g.add_edge(
                    str(r1.id),
                    str(r2.id),
                    weight=1,
                    color=REACTION_GRAPH_COLOR,
                    alpha=0,
                )

    return g


def omit_single_direction_metabolites(g: nx.DiGraph) -> None:
    """Omit single direction metabolites from the graph."""
    to_remove = set()
    for node in g.nodes:
        if g.nodes[node]["shape"] == METABOLITE_SHAPE:
            in_degree = g.in_degree(node)
            out_degree = g.out_degree(node)
            if in_degree == 0 or out_degree == 0:
                to_remove.add(node)

    if to_remove:
        print(
            f"Omitting the following single direction metabolites: {', '.join(to_remove)}"
        )
        g.remove_nodes_from(to_remove)


def differentiate_highly_variable_genes(
    g: nx.DiGraph, meta_pert_data: MetaPertDataset
) -> None:
    """Differentiate highly variable genes in the graph."""
    highly_variable_genes = set(
        meta_pert_data.adata.var.index[meta_pert_data.adata.var["highly_variable"]]
    )
    for node in g.nodes:
        if g.nodes[node]["shape"] == GENE_SHAPE and node in highly_variable_genes:
            g.nodes[node]["color"] = HIGHLY_VARIABLE_GENE_COLOR
            for successor in g.successors(node):
                g.edges[node, successor]["color"] = HIGHLY_VARIABLE_GENE_COLOR


def differentiate_perturbed_genes(
    g: nx.DiGraph, meta_pert_data: MetaPertDataset
) -> None:
    """Differentiate perturbed genes in the graph."""
    perturbed_genes = set(meta_pert_data.perturbed_genes_list)
    for node in g.nodes:
        if g.nodes[node]["shape"] == GENE_SHAPE and node in perturbed_genes:
            g.nodes[node]["bbox"] = dict(
                facecolor="none",
                edgecolor=PERTURBED_GENE_EDGE_COLOR,
                boxstyle="round,pad=0.3",
            )


def apply_plot_reaction_graph(g: nx.DiGraph) -> None:
    """Apply plot reaction graph settings."""
    for node in g.nodes:
        if g.nodes[node]["shape"] == REACTION_SHAPE:
            g.nodes[node]["color"] = REACTION_GRAPH_COLOR
            g.nodes[node]["alpha"] = 0.8

    for node in g.nodes:
        if g.nodes[node]["shape"] == METABOLITE_SHAPE:
            g.nodes[node]["alpha"] = 0.1

    for u, v, d in g.edges(data=True):
        # If both nodes are not a gene and a reaction, set alpha to 0.1
        if g.nodes[u]["shape"] != GENE_SHAPE and g.nodes[v]["shape"] != REACTION_SHAPE:
            d["alpha"] = 0.1
        if d["color"] == REACTION_GRAPH_COLOR:
            # If the edge is a reaction graph edge, set alpha to 0.5
            d["alpha"] = 0.5
        if (
            # If the edge is between a metabolite and a reaction, set alpha to 0.2
            g.nodes[u]["shape"] == METABOLITE_SHAPE
            and g.nodes[v]["shape"] == REACTION_SHAPE
        ):
            d["alpha"] = 0.1


def normalize_scores(scores):
    """Normalize scores to a range between 0 and 1."""
    min_score = min(scores.values())
    max_score = max(scores.values())
    return {k: (v - min_score) / (max_score - min_score) for k, v in scores.items()}


def differentiate_gene_expression_mean(
    g: nx.DiGraph, meta_pert_data: MetaPertDataset, cmap
) -> None:
    """Differentiate gene expression mean in the graph."""
    if "means" not in meta_pert_data.adata.var:
        print("Pre-calculated means not available in adata.var.")
        return

    gene_expression_means = meta_pert_data.adata.var["means"].to_dict()
    norm_scores = normalize_scores(gene_expression_means)

    for node in g.nodes:
        if g.nodes[node]["shape"] == GENE_SHAPE and node in norm_scores:
            g.nodes[node]["color"] = mcolors.to_hex(cmap(norm_scores[node]))


def differentiate_reaction_activation_mean(
    g: nx.DiGraph, meta_pert_data: MetaPertDataset, cmap
) -> None:
    """Differentiate reaction activation mean in the graph."""
    if (
        not hasattr(meta_pert_data, "reaction_stats")
        or "mean" not in meta_pert_data.reaction_stats
    ):
        print("Reaction mean statistics not available.")
        return

    mean_scores = meta_pert_data.reaction_stats["mean"]
    norm_scores = normalize_scores(mean_scores)

    for node in g.nodes:
        if g.nodes[node]["shape"] == REACTION_SHAPE and node.upper() in norm_scores:
            g.nodes[node]["color"] = mcolors.to_hex(cmap(norm_scores[node.upper()]))


def differentiate_gene_expression_variance(
    g: nx.DiGraph, meta_pert_data: MetaPertDataset, cmap
) -> None:
    """Differentiate gene expression variance in the graph."""
    if "dispersions" not in meta_pert_data.adata.var:
        print("Pre-calculated dispersions not available in adata.var.")
        return

    gene_expression_variances = meta_pert_data.adata.var["dispersions"].to_dict()
    norm_scores = normalize_scores(gene_expression_variances)

    for node in g.nodes:
        if g.nodes[node]["shape"] == GENE_SHAPE and node in norm_scores:
            g.nodes[node]["color"] = mcolors.to_hex(cmap(norm_scores[node]))


def differentiate_reaction_activation_variance(
    g: nx.DiGraph, meta_pert_data: MetaPertDataset, cmap
) -> None:
    """Differentiate reaction activation variance in the graph."""
    if (
        not hasattr(meta_pert_data, "reaction_stats")
        or "variance" not in meta_pert_data.reaction_stats
    ):
        print("Reaction variance statistics not available.")
        return

    variance_scores = meta_pert_data.reaction_stats["variance"]
    norm_scores = normalize_scores(variance_scores)

    for node in g.nodes:
        if g.nodes[node]["shape"] == REACTION_SHAPE and node.upper() in norm_scores:
            g.nodes[node]["color"] = mcolors.to_hex(cmap(norm_scores[node.upper()]))


def draw_nodes_and_edges(g: nx.DiGraph, pos) -> None:
    """Draw nodes and edges of the graph."""
    node_shapes = {METABOLITE_SHAPE: "o", GENE_SHAPE: "s", REACTION_SHAPE: "d"}
    for shape in node_shapes:
        nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=[n for n in g.nodes if g.nodes[n]["shape"] == shape],
            node_shape=node_shapes[shape],
            node_color=[
                g.nodes[n]["color"] for n in g.nodes if g.nodes[n]["shape"] == shape
            ],
            label=[
                g.nodes[n]["label"] for n in g.nodes if g.nodes[n]["shape"] == shape
            ],
            node_size=1000,
            alpha=[
                g.nodes[n].get("alpha", 1.0)
                for n in g.nodes
                if g.nodes[n]["shape"] == shape
            ],
        )

    edges = g.edges(data=True)
    for u, v, d in edges:
        edge_alpha = d.get("alpha", 0.5)
        nx.draw_networkx_edges(
            g,
            pos,
            edgelist=[(u, v)],
            arrowstyle="-|>",
            arrowsize=30,
            width=2,
            alpha=edge_alpha,
            edge_color=d["color"],
        )

    edge_labels = {(u, v): d["weight"] for u, v, d in edges}
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=8)

    # Draw node labels with bounding boxes if specified
    for node, (x, y) in pos.items():
        label = g.nodes[node]["label"]
        bbox = g.nodes[node].get("bbox", None)
        plt.text(
            x,
            y,
            label,
            fontsize=10,
            bbox=bbox,
            ha="center",
            va="center",
        )


def plot_single_subgraph(
    subgraph: nx.DiGraph,
    pos,
    meta_pert_data: MetaPertDataset,
    subsystem_name: str,
    base_figsize: tuple,
    cmap,
    differentiate_reaction_activation_mean_flag: bool,
    differentiate_reaction_activation_variance_flag: bool,
    differentiate_gene_expression_mean_flag: bool,
    differentiate_gene_expression_variance_flag: bool,
    component: str = "",
):
    """Plot a single subgraph."""
    num_nodes = len(subgraph.nodes)
    figsize = (
        base_figsize[0] + num_nodes * 0.3,
        base_figsize[1] + num_nodes * 0.3,
    )

    fig, ax = plt.subplots(figsize=figsize)

    if (
        differentiate_reaction_activation_mean_flag
        or differentiate_reaction_activation_variance_flag
        or differentiate_gene_expression_mean_flag
        or differentiate_gene_expression_variance_flag
    ):
        # Add a single colorbar for the entire plot
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        plt.colorbar(
            sm,
            ax=ax,
            orientation="horizontal",
            label="Normalized Scores",
            shrink=0.2,
            aspect=30,
            location="top",
        )

    if differentiate_reaction_activation_mean_flag:
        differentiate_reaction_activation_mean(subgraph, meta_pert_data, cmap)

    if differentiate_reaction_activation_variance_flag:
        differentiate_reaction_activation_variance(subgraph, meta_pert_data, cmap)

    if differentiate_gene_expression_mean_flag:
        differentiate_gene_expression_mean(subgraph, meta_pert_data, cmap)

    if differentiate_gene_expression_variance_flag:
        differentiate_gene_expression_variance(subgraph, meta_pert_data, cmap)

    draw_nodes_and_edges(subgraph, pos)
    plt.title(f"Subsystem: {subsystem_name} {component}")
    plt.axis("off")
    plt.show()


def plot_subsystem_graph(
    subsystem_name: str,
    meta_pert_data: MetaPertDataset,
    base_figsize: tuple = (10, 10),
    plot_by_connected_component: bool = False,
    omit_single_direction_metabolites_flag: bool = False,
    differentiate_highly_variable_genes_flag: bool = False,
    differentiate_perturbed_genes_flag: bool = False,
    differentiate_reaction_activation_mean_flag: bool = False,
    differentiate_reaction_activation_variance_flag: bool = False,
    differentiate_gene_expression_mean_flag: bool = False,
    differentiate_gene_expression_variance_flag: bool = False,
    plot_reaction_graph_flag: bool = False,
):
    """Plot a directed weighted graph of the given subsystem."""
    g = build_basic_subsystem_graph(subsystem_name, meta_pert_data)

    if omit_single_direction_metabolites_flag:
        omit_single_direction_metabolites(g)

    if differentiate_highly_variable_genes_flag:
        differentiate_highly_variable_genes(g, meta_pert_data)

    if differentiate_perturbed_genes_flag:
        differentiate_perturbed_genes(g, meta_pert_data)

    if plot_reaction_graph_flag:
        apply_plot_reaction_graph(g)

    cmap = GRADIENT_MAP

    plot_args = [
        meta_pert_data,
        subsystem_name,
        base_figsize,
        cmap,
        differentiate_reaction_activation_mean_flag,
        differentiate_reaction_activation_variance_flag,
        differentiate_gene_expression_mean_flag,
        differentiate_gene_expression_variance_flag,
    ]

    if plot_by_connected_component:
        for i, component in enumerate(nx.weakly_connected_components(g)):
            # if the number of nodes in the component is less than 5, skip plotting
            if len(component) < 5:
                continue
            subgraph = g.subgraph(component)
            pos = nx.nx_agraph.graphviz_layout(subgraph, prog="dot")
            plot_single_subgraph(
                subgraph,
                pos,
                *plot_args,
                component=f"- Component {i + 1}",
            )
    else:
        pos = nx.nx_agraph.graphviz_layout(g, prog="dot")
        plot_single_subgraph(
            g,
            pos,
            *plot_args,
        )
