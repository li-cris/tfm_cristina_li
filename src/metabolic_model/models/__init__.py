"""This module provides functions to load and initialize metabolic models.

Original code from the Compass (https://github.com/YosefLab/Compass) tool.

BSD 3-Clause License

Copyright (c) 2020, YosefLab All rights reserved.
"""

from __future__ import absolute_import, division, print_function

import os

import libsbml
from globals import EXCHANGE_LIMIT, MODEL_DIR

from . import import_matlab, import_sbml2, import_sbml3
from .gene_symbols import convert_species, resolve_genes
from .import_common import clean_reactions, limit_maximum_flux
from .metabolic_model_transmet import MetabolicModelTransmet

# ----------------------------------------
# Loading models from either XML or MATLAB outputs
# ----------------------------------------


def load_metabolic_model(model_name, species="homo_sapiens"):
    """Load the metabolic model from `file_name`, returning a Model object."""
    if model_name.endswith("_mat"):
        model = import_matlab.load(model_name, species)
    else:
        model_dir = os.path.join(MODEL_DIR, model_name)
        model_file = [
            x
            for x in os.listdir(model_dir)
            if x.lower().endswith(".xml") or x.lower().endswith(".xml.gz")
        ]

        if len(model_file) == 0:
            raise Exception(
                "Invalid model - could not find .xml or .xml.gz file in " + model_dir
            )
        else:
            model_file = model_file[0]

        full_path = os.path.join(model_dir, model_file)
        sbml_document = libsbml.readSBMLFromFile(full_path)

        level = sbml_document.getLevel()

        if level == 3:
            model = import_sbml3.load(model_name, sbml_document)
        elif level == 2:
            model = import_sbml2.load(model_name, sbml_document)
        else:
            raise Exception("Invalid level {} for model {}".format(level, model_file))

        resolve_genes(model)
        convert_species(model, species)
        clean_reactions(model)
        limit_maximum_flux(model, 1000)

    return model


def init_model(model, species, exchange_limit, media=None, isoform_summing="legacy"):
    """Initialize a metabolic model for simulation."""
    model = load_metabolic_model(model, species)

    # Limit exchange reactions
    model.limitExchangeReactions(limit=exchange_limit)

    # Split fluxes into _pos / _neg
    model.make_unidirectional()

    if media is not None:
        model.load_media(media)

    if isoform_summing == "remove-summing":
        model.remove_isoform_summing()

    return model


def init_model_transmet(model_name: str, **kwargs) -> None:
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
        "exchange_limit": EXCHANGE_LIMIT,
    }

    # Update default arguments with any provided kwargs
    default_args.update(kwargs)

    print(f"Initializing metabolic model '{model_name}' with the following parameters:")
    for key, value in default_args.items():
        print(f"  {key}: {value}")

    compass_model = init_model(
        model=model_name,
        species=default_args["species"],
        exchange_limit=default_args["exchange_limit"],
        media=default_args["media"],
        isoform_summing=default_args["isoform_summing"],
    )

    # Create an instance of MetabolicModelTransmet and populate it with the data
    model = MetabolicModelTransmet(model_name, compass_model=compass_model)

    return model
