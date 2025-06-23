# Systematic Perturbation Prediction

## Linear Gene Expression Model

Train and test two versions of the linear gene expression model ("optimized" and "learned"):

```shell
python3 -m src.lgem.main
```

## ToDo

- Add Dockerfiles for tools: SENA, scgpt and GEARS

[`src/data_utils`]
- (Loss functions moved here)
- Add code for prediction overview when finished

[`src/gears_tools`]
- Update other tools' code that uses GEARS pert_data data splitter and loader to follow the recent change.

[`src/scgpt_tools`]
- Seems to work fine in low GPU workspaces if embsize is kept as half the default one and dataloader batch sizes are decreased.

[`src/lgem`]
- Clean up and merge some scripts currently unused.

[`src/sena`]
- See how to manage problems related to using functions from SENA repo
