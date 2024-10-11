"""Import a metabolic model from a Matlab model."""

from __future__ import absolute_import, division, print_function

import json
import os

from globals import MODEL_DIR

from .import_matlab import load as load_common
from .metabolic_model_transmet import MetabolicModelTransmet, Subsystem


def load_transmet(model_name, species):
    """model_name: str.

        Name of the folder containing the model
    species: str
        Species name.  either 'homo_sapiens' or 'mus_musculus'
    """
    # Use the common load function to get the data
    common_model = load_common(model_name, species)

    # Create an instance of MetabolicModelTransmet and populate it with the data
    model = MetabolicModelTransmet(model_name)
    model.reactions = common_model.reactions
    model.species = common_model.species

    # Load subsystems specific to MetabolicModelTransmet
    top_dir = os.path.join(MODEL_DIR, model_name)
    model_dir = os.path.join(top_dir, "model")

    with open(os.path.join(model_dir, "model.subSystems.json")) as fin:
        subsystems = json.load(fin)

    groups = zip(common_model.reactions.values(), subsystems)
    for reaction, subsystem in groups:
        if subsystem not in model.subsystems:
            model.subsystems[subsystem] = Subsystem(name=subsystem)
        model.subsystems[subsystem].add_reaction(reaction)

    return model
