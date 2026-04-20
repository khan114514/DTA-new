# DTA-new

Minimal regression code snapshot extracted from `MGraphDTA`, focused on the
`full_model` branch implementation and training pipeline.

Included files:

- `regression/model.py`
- `regression/engine.py`
- `regression/train.py`
- `regression/test.py`
- supporting regression utilities required to run the model

Key fixes in this snapshot:

- `quantity_branch` is now structurally independent from `interaction_prior`
- branch decorrelation in `full_model` now acts on the actual mechanism
  branches instead of encoder outputs
- quantity auxiliary loss is only enabled when a separate `quantity_target`
  exists in the batch
- validation and checkpoint selection now use the main affinity loss
- training returns to `train()` after validation and uses real epoch semantics
