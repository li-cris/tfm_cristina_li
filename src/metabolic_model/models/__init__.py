"""This module provides functions to load and initialize metabolic models.

Original code from the Compass (https://github.com/YosefLab/Compass) tool.

BSD 3-Clause License

Copyright (c) 2020, YosefLab All rights reserved.
"""

from __future__ import absolute_import, division, print_function

import os

import libsbml
from globals import MODEL_DIR

from . import import_matlab_transmet, import_sbml2, import_sbml3
from .gene_symbols import convert_species, resolve_genes
from .import_common import clean_reactions, limit_maximum_flux

# ----------------------------------------
# Loading models from either XML or MATLAB outputs
# ----------------------------------------


def load_metabolic_model(model_name, species="homo_sapiens"):
    """Load the metabolic model from `file_name`, returning a Model object."""
    if model_name.endswith("_mat"):
        model = import_matlab_transmet.load_transmet(model_name, species)
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
