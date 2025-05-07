# Systematic Perturbation Prediction

## Linear Gene Expression Model

Train and test two versions of the linear gene expression model ("optimized" and "learned"):

```shell
python3 -m src.lgem.main
```

## ToDo

[`src/data_utils`]
- Manage module calling from different scripts

[`src/gears_tools`]
- Move loss functions outside the folder, change module calling

[`src/scgpt`]
- Solve issue with memory allocation during evaluation

[`src/lgem`]
- Merge altered model into main.py and train.py

[`src/sena`]