"""Global variables for use by other modules."""

import os

_this_directory = os.path.dirname(os.path.abspath(__file__))

GIT_DIR = os.path.abspath(os.path.join(_this_directory, "..", ".git"))

RESOURCE_DIR = os.path.join(_this_directory, "resources")
MODEL_DIR = os.path.join("resources", "Metabolic Models")

# Parameters for Compass
EXCHANGE_LIMIT = 1.0  # Limit for exchange reactions
