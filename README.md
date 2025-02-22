# Systematic Perturbation Prediction

## Linear Gene Expression Model

Train and test two versions of the linear gene expression model ("optimized" and "learned"):

```shell
python3 -m src.lgem.main
```

## Cristina's Todos

[`src/gears/mmd_loss.py`](src/gears/mmd_loss.py):
- Improve.
- Can it only be used with batches?
- Make similar functionality for other metrics.

[`src/gears/train_predict_evaluate.py`](src/gears/train_predict_evaluate.py):
- Run, understand, and improve.
- Extend to other tools (framework?).
