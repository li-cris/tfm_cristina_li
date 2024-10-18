"""For working with metabolic models."""

from __future__ import absolute_import, division, print_function

import json

# Load subsystems specific to MetabolicModelTransmet
import os

from globals import MODEL_DIR

from metabolic_model.models.gene_symbols_transmet import (
    # TODO change function name
    convert_gene_symbols_to_ensembl_ids,
)
from metabolic_model.models.metabolic_model import MetabolicModel

# ----------------------------------------
# Model class and related classes
# ----------------------------------------


class MetabolicModelTransmet(MetabolicModel):
    """A class representing a metabolic model."""

    def __init__(self, name, compass_model=None):
        """Initialize the metabolic model.

        Args:
            name (str): The name of the metabolic model.
            compass_model (object, optional): The compass model to initialize from.
        """
        super().__init__(name)
        self.subsystems = {}

        if compass_model is not None:
            self.reactions = compass_model.reactions
            self.species = compass_model.species

            top_dir = os.path.join(MODEL_DIR, self.name)
            model_dir = os.path.join(top_dir, "model")

            with open(os.path.join(model_dir, "model.subSystems.json")) as fin:
                subsystems = json.load(fin)

            groups = zip(compass_model.reactions.values(), subsystems)
            for reaction, subsystem in groups:
                if subsystem not in self.subsystems:
                    self.subsystems[subsystem] = Subsystem(name=subsystem)
                self.subsystems[subsystem].add_reaction(reaction)

            print("Metabolic model initialized successfully.")

            # Remove empty gene associations
            print("Removing empty gene associations...")
            self.remove_empty_gene_associations()
            print("Empty gene associations removed.")

            # Convert gene symbols to ensembl ids
            print("Converting gene symbols to Ensembl IDs...")
            self.convert_gene_symbols_to_ensembl_ids()
            print("Gene symbols converted to Ensembl IDs.")

            print("Metabolic model loading complete.")

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
