# GRAD-SCOPE

Instrumenting **gradient-flow dynamics in deep CNNs** to study when and why
training degrades — and how early per-layer gradient signals warn of it.

## Idea

Attach a lightweight logger to a training run, record per-layer gradient
statistics every step, classify each layer by type and depth, then measure the
**lead time** between gradient-flow anomalies (vanishing / exploding / collapse)
and an observable drop in validation performance.

## Structure

```
models/          # architectures under study
  resnet20.py    # CIFAR-style ResNet-20
  vgg11.py       # CIFAR-style VGG-11
logger/          # gradient instrumentation
  gradient_logger.py   # per-layer gradient statistics via hooks
  layer_classifier.py  # layer type + normalized depth
  lead_time.py         # anomaly -> performance-drop lead time
experiments/     # experiment configs and runners
  baseline.py          # reference training run with logging
  run_experiment.py    # CLI entry point
results/         # logged metrics and records (gitignored contents)
figures/         # generated plots (gitignored contents)
```

## Install

```sh
pip install -r requirements.txt
```

## Run

```sh
python -m experiments.run_experiment
```

> Status: scaffolding only — files are stubs, no implementation yet.
