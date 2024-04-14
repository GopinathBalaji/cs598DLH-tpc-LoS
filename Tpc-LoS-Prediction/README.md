# Experiment Configuration and Code Changes

## Batch Size Adjustment for CUDA Error

To address the CUDA memory allocation error (`CUBLAS_STATUS_ALLOC_FAILED`), the batch size in the configuration was reduced to manage GPU memory usage effectively.

### Changes Made:

- Batch size(`c['batch_size']`) reduced from 512 to 256 in the `models\final_experiment_scripts\best_hyperparameters.py` under `best_lstm` method.

## Batch Size Check to Avoid Batch Normalization Error

A check was implemented to skip batches of size 1 during testing to prevent errors with batch normalization layers, which require more than one value per channel.

### Changes Made:

- Added a conditional check in the `test` method of the `ExperimentTemplate` class to skip batches if `batch[0].size(0) == 1`(line 412-413).

## Experiments Run

The following experiments have been run so far by using the scripts in the `models\final_experiment_scripts`:

- `MIMIC.LoS.tpc`
- `MIMIC.multitask.tpc`
- `eICU.LoS.tpc6p25` with batch size check.
- `eICU.LoS.channel_wise_lstm6p25` with batch size adjustment.

Please check the corresponding `models\experiments\final\<dataset>\<task>\<model>\` folder to view the metrics and results.
