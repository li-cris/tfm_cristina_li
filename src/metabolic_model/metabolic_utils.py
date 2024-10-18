"""Utilities for metabolic models."""

import numpy as np


def get_reaction_consistencies(compass_reaction_penalties, min_range=1e-3):
    """Convert the raw penalties outputs of compass into scores per reactions.

    Higher numbers indicate more activity.
    """
    df = -np.log(compass_reaction_penalties + 1)
    df = df[df.max(axis=1) - df.min(axis=1) >= min_range]
    df = df - df.min().min()
    return df


def print_subsystems_stats(metabolic_model):
    """Print statistics about the subsystems."""
    # Total number of subsystems
    total_subsystems = len(metabolic_model.subsystems)

    # Number of reactions per subsystem
    num_reactions_per_subsystem = {
        subsystem.name: len(subsystem.get_reactions())
        for subsystem in metabolic_model.subsystems.values()
    }

    # Number of genes per subsystem
    num_genes_per_subsystem = {
        subsystem.name: len(subsystem.get_associated_genes())
        for subsystem in metabolic_model.subsystems.values()
    }

    # Calculate average number of reactions per subsystem
    total_reactions = sum(num_reactions_per_subsystem.values())
    avg_reactions_per_subsystem = (
        total_reactions / total_subsystems if total_subsystems > 0 else 0
    )

    # Calculate average number of genes per subsystem
    total_genes = sum(num_genes_per_subsystem.values())
    avg_genes_per_subsystem = (
        total_genes / total_subsystems if total_subsystems > 0 else 0
    )

    # Find the subsystem with the maximum number of reactions
    max_reactions_subsystem = max(
        num_reactions_per_subsystem, key=num_reactions_per_subsystem.get
    )
    max_reactions = num_reactions_per_subsystem[max_reactions_subsystem]

    # Find the subsystem with the minimum number of reactions
    min_reactions_subsystem = min(
        num_reactions_per_subsystem, key=num_reactions_per_subsystem.get
    )
    min_reactions = num_reactions_per_subsystem[min_reactions_subsystem]

    # Find the subsystem with the maximum number of genes
    max_genes_subsystem = max(num_genes_per_subsystem, key=num_genes_per_subsystem.get)
    max_genes = num_genes_per_subsystem[max_genes_subsystem]

    # Find the subsystem with the minimum number of genes
    min_genes_subsystem = min(num_genes_per_subsystem, key=num_genes_per_subsystem.get)
    min_genes = num_genes_per_subsystem[min_genes_subsystem]

    # Print the statistics as a report
    print("Subsystems Statistics Report")
    print("=" * 40)
    print(f"Total number of subsystems: {total_subsystems}")
    print("-" * 40)
    print(
        f"Average number of reactions per subsystem: {avg_reactions_per_subsystem:.2f}"
    )
    print(
        f"Subsystem with the maximum reactions: '{max_reactions_subsystem}' "
        f"with {max_reactions} reactions"
    )
    print(
        f"Subsystem with the minimum reactions: '{min_reactions_subsystem}' "
        f"with {min_reactions} reactions"
    )
    print("-" * 40)
    print(f"Average number of genes per subsystem: {avg_genes_per_subsystem:.2f}")
    print(
        f"Subsystem with the maximum genes: '{max_genes_subsystem}' "
        f"with {max_genes} genes"
    )
    print(
        f"Subsystem with the minimum genes: '{min_genes_subsystem}' "
        f"with {min_genes} genes"
    )
    print("=" * 40)


def print_dataset_metabolic_info(
    meta_pert_data, top_n_subsystems=10, variance_type="dispersions"
) -> None:
    """Print information about the metabolic genes in the perturbation dataset."""
    print("Metabolic Model Information Report")
    print("=" * 40)

    # Use the saved perturbed_genes_list from the MetaPertDataset instance
    perturbed_genes = meta_pert_data.perturbed_genes_list

    # Get the metabolic genes of the dataset.
    metabolic_model_genes = meta_pert_data.metabolic_model.get_genes()

    # Get the dataset genes from the index
    dataset_genes = meta_pert_data.adata.var.index.tolist()

    # Get the metabolic genes that are in the dataset.
    metabolic_genes_in_dataset = [
        gene for gene in dataset_genes if gene in metabolic_model_genes
    ]

    # Print the number of metabolic genes in the dataset.
    print(
        f"Number of metabolic genes in the dataset: "
        f"{len(metabolic_genes_in_dataset)}"
    )

    # Calculate the number of metabolic genes in the dataset that are perturbed
    perturbed_metabolic_genes_in_dataset = [
        gene for gene in metabolic_genes_in_dataset if gene in perturbed_genes
    ]
    print(
        f"Number of metabolic genes in the dataset that are perturbed: "
        f"{len(perturbed_metabolic_genes_in_dataset)}"
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

    # Calculate the number of metabolic genes among the most variable genes that are perturbed
    perturbed_metabolic_genes_in_most_variable = [
        gene for gene in metabolic_genes_in_most_variable if gene in perturbed_genes
    ]
    print(
        f"Number of metabolic genes among the most variable genes that are perturbed: "
        f"{len(perturbed_metabolic_genes_in_most_variable)}"
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

    # Calculate subsystems with metabolic genes in dataset
    subsystems_with_metabolic_genes_in_dataset = [
        subsystem
        for subsystem in meta_pert_data.metabolic_model.subsystems.values()
        if any(
            gene in metabolic_genes_in_dataset
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

    # Print the number of subsystems with metabolic genes in dataset
    print(
        f"Number of subsystems with metabolic genes in the dataset: "
        f"{len(subsystems_with_metabolic_genes_in_dataset)}/{total_subsystems}"
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

        # Calculate the number of perturbed genes in the subsystem
        num_perturbed_genes = sum(gene in perturbed_genes for gene in subsystem_genes)

        print(
            f"Subsystem: {subsystem_name}, Variance: {variance:.2f}, "
            f"Percentage of metabolic variance: {variance_percentage:.2f}%, "
            f"Genes in Most Variable: {num_genes_in_most_variable}/{total_genes}, "
            f"Perturbed Genes: {num_perturbed_genes}/{total_genes}"
        )

    # Print the total percentage of variance captured by the top N subsystems
    top_n_total_variance = sum(variance for _, variance in top_subsystems)
    top_n_variance_percentage = (top_n_total_variance / total_metabolic_variance) * 100
    print(
        f"\nTotal percentage of metabolic variance captured by the top "
        f"{top_n_subsystems} subsystems: {top_n_variance_percentage:.2f}%"
    )
    print("=" * 40)


def export_variable_perturbed_subsystems(
    meta_pert_data,
    top_n_subsystems=10,
    variance_type="dispersions",
    output_file="top_subsystems.txt",
) -> None:
    """Export the top subsystems with the highest accumulated variance to a file.

    Args:
        meta_pert_data (MetaPertDataset): The MetaPertDataset object.
        top_n_subsystems (int): The number of top subsystems to export.
        variance_type (str): The type of variance to consider.
        output_file (str): The name of the output file.
    """
    # Use the saved perturbed_genes_list from the MetaPertDataset instance
    perturbed_genes = meta_pert_data.perturbed_genes_list

    # Get the metabolic genes of the dataset.
    metabolic_model_genes = meta_pert_data.metabolic_model.get_genes()

    # Identify the most variable genes
    most_variable_genes = meta_pert_data.adata.var[
        meta_pert_data.adata.var["highly_variable"]
    ].index.tolist()

    # Get the metabolic genes that are among the most variable genes
    metabolic_genes_in_most_variable = [
        gene for gene in most_variable_genes if gene in metabolic_model_genes
    ]

    # Fill NaN values with zero to avoid numerical problems
    meta_pert_data.adata.var[variance_type] = meta_pert_data.adata.var[
        variance_type
    ].fillna(0)

    # Calculate subsystems with metabolic genes in most variable genes
    subsystems_with_metabolic_genes_in_most_variable = [
        subsystem
        for subsystem in meta_pert_data.metabolic_model.subsystems.values()
        if any(
            gene in metabolic_genes_in_most_variable
            for gene in subsystem.get_associated_genes()
        )
    ]

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

    # Export the top N subsystems to the output file
    with open(output_file, "w") as f:
        for subsystem_name, variance in top_subsystems:
            f.write(f"{subsystem_name}\n")

    # Print the exported subsystems with their variance and perturbed genes
    print(f"\nExported Top {top_n_subsystems} Subsystems to {output_file}:")
    print("-" * 40)
    for subsystem_name, variance in top_subsystems:
        subsystem = meta_pert_data.metabolic_model.subsystems[subsystem_name]
        subsystem_genes = subsystem.get_associated_genes()
        num_perturbed_genes = sum(gene in perturbed_genes for gene in subsystem_genes)
        print(
            f"Subsystem: {subsystem_name}, Variance: {variance:.2f}, "
            f"Perturbed Genes: {num_perturbed_genes}/{len(subsystem_genes)}"
        )
