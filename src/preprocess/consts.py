AVAIL_DATASETS = [
    "adamson",
    "dixit",
    "norman",
    "replogle_k562_essential",
    "replogle_rpe1_essential",
]

DATASET_TO_GEOID = {
    "adamson": "GSE90546",
    "dixit": "GSE90063",
    "norman": "GSE133344",
}

DATASET_TO_GDRIVE_URL = {
    "adamson": "https://drive.google.com/uc?id=1W1phErDoQ9U5iJZSEuyEZM4dF8U8ZWqf",
    "dixit": "https://drive.google.com/uc?id=1BN6gwKFgJIpR9fXfdQ9QeHm8mAzvmhKQ",
    "norman": "https://drive.google.com/uc?id=1T5_varFOGWUtSig4RQRCSsfPxivUwd9j",
    "replogle_k562_essential": "https://drive.google.com/uc?id=12flxmpj-XnJ8BZKtf-sgBhdUN2X4v7CD",
    "replogle_rpe1_essential": "https://drive.google.com/uc?id=1b-ZwE_Y6dNKqb4KQgUgFKfl6OGC8lmYE",
}

GEARS_OBS_FNAME = "obs.csv"

GZ_FNAME = "adata.h5ad.gz"
H5AD_FNAME = "adata.h5ad"
GENE_EXP_FNAME = "expression.tsv"

CELL_ID_COLNAME = "cell_id"

#? 10x Genomic related naming scheme
EXP_NAME_COLNAME = 'exp_name'
EXP_CFG_COLNAME = 'exp_config'