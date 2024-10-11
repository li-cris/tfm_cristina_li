"""For working with metabolic models."""

from __future__ import absolute_import, division, print_function

from metabolic_model.models.gene_symbols_transmet import (
    convert_gene_symbols_to_ensembl_ids,
)
from metabolic_model.models.metabolic_model import MetabolicModel

# ----------------------------------------
# Model class and related classes
# ----------------------------------------


class MetabolicModelTransmet(MetabolicModel):
    """A class representing a metabolic model."""

    def __init__(self, name):
        """Initialize a MetabolicModel instance.

        Args:
            name (str): The name of the metabolic model.
        """
        super().__init__(name)
        self.subsystems = {}

    def get_genes(self):
        """Return a list of gene id's in the MetabolicModel."""
        genes = set()
        for reaction in self.reactions.values():
            genes.update(reaction.list_genes())

        return list(genes)

    def get_subsystems(self):
        """Return a list of subsystems in the MetabolicModel."""
        return list(self.subsystems.keys())

    def remove_empty_gene_associations(self):
        """Remove gene associations and genes if the gene name is empty."""

        def update_association(assoc):
            if assoc.type == "gene":
                if assoc.gene.name == "":
                    return None
            else:
                assoc.children = [
                    update_association(child)
                    for child in assoc.children
                    if update_association(child) is not None
                ]
                if not assoc.children:
                    return None
            return assoc

        for reaction in self.reactions.values():
            if reaction.gene_associations is not None:
                reaction.gene_associations = update_association(
                    reaction.gene_associations
                )
                if reaction.gene_associations is None:
                    reaction.gene_associations = None

    def convert_gene_symbols_to_ensembl_ids(self):
        """Convert all gene symbols or names in the model to Ensembl IDs."""
        convert_gene_symbols_to_ensembl_ids(self)

    def __str__(self) -> str:
        """Return a string representation of the MetabolicModelTransmet object."""
        total_subsystems = len(self.subsystems)
        total_reactions = len(self.reactions)
        total_genes = len(self.get_genes())

        return (
            f"MetabolicModelTransmet object\n"
            f"    name: {self.name}\n"
            f"    Total number of subsystems: {total_subsystems}\n"
            f"    Total number of reactions: {total_reactions}\n"
            f"    Total number of associated genes: {total_genes}"
        )


class Subsystem(object):
    """A class representing a subsystem in a metabolic model."""

    def __init__(self, name=None):
        """Initialize a Subsystem instance."""
        self.name = name
        self.reactions = []

    def add_reaction(self, reaction):
        """Add a reaction to the subsystem."""
        self.reactions.append(reaction)

    def get_reactions(self):
        """Return a list of reaction id's in the Subsystem."""
        return [x.id for x in self.reactions]

    def get_associated_genes(self):
        """Return a list of genes associated with the Subsystem."""
        genes = set()
        for reaction in self.reactions:
            genes.update(reaction.list_genes())

        return list(genes)

    def print_subsystem_info(self):
        """Print information about subsystem."""
        print(self.name)
        # Print number of reactions
        print("Number of associated reactions: ", len(self.reactions))
        # Print number of associated genes
        print("Number of associated genes: ", len(self.get_associated_genes()))

    def to_serializable(self):
        """Convert the Subsystem object to a serializable dictionary format."""
        return {"id": self.id, "name": self.name}


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
