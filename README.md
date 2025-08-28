# Systematic Review of Performance Metrics for Gene Expression Data using Perturb-Seq datasets as a Benchmark
This GitHub repository contains:
- the source code for training and evaluation
- some demo Jupyter Notebooks depicting the process with a reduced dataset
- Jupyter Notebooks used for graph-related tasks
</br>

**IMPORTANT**: These models are recommended to be ran in CUDA-available environments.
</br>

Dockerfiles for GEARS (as well as the linear models and graph generation) and scGPT are available in ```/dockerfiles```. </br>
The Dockerfile for SENA is available at https://github.com/ML4BM-Lab/SENA. Follow the process outlined on the main page to set up the repository and download the given full Norman2019_raw dataset as well. </br>
Be aware that these docker images are quite resource-heavy (over 10 GB each). </br>
</br>
If you want to use this code, clone this repository: </br>

```bash
git clone --branch develop-cris https://github.com/voges/tfm_cristina_li.git
```

And it’s recommended to keep the work in a directory parallel to the cloned repository. </br>

```
your-parent-folder/
├── project_dir/       ← your own work
└── tfm_cristina_li/         ← cloned repository
```

Ideally, project_dir should have a data/, models/ and results/ subdirectories.

```
project_dir/
├── data/       ← where to keep scRNA-seq data
├── models/       ← your trained models and scGPT foundation model
└── results/       ← your metrics and mean predicitons
```

## Before starting
### For scGPT (IMPORTANT)
Download the full pre-trained **whole_human** foundation model from [scGPT repository](https://github.com/bowang-lab/scGPT/tree/main), and keep it in ```/models/```.

### Perturb-Seq datasets
To get ReplogleRPE1 dataset, GEARS’ own data loader was used (source of dataset: https://dataverse.harvard.edu/api/access/datafile/7458694).
To get the alternative Norman2019 dataset for scGPT and linear models, run the script ‘SCRIPT_NAME’ from project_dir. This same script can be used to get the corresponding dataset for ReplogleRPE1 with perturbation conditions found in the featured genes.
