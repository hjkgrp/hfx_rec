The files contained in this directory are:

- The `RACs` folder, containing `train.py`, `csd_racs_predictions.csv`, and `vss_racs_predictions.csv`
  - The `train.py` script trains all of the RACs-based ML models tested. The `.csv` files contain the resultant train/val/test splits in the case of VSS-452-HFX and the predictions for every model tested.
- The `DF-BP` folder:
  - Each subfolder is named as `<functional-for-density>_opt<seed>`, where the folders ending in `dfarec` indicate that the train/val split was done as to mimic the splits in the DFA recommender work.
  - In each subfolder, there are several files: `BP_model-<functional>.pkl`, a checkpoint of the best-performing model for either PBEx or SCANx, with the suffix differentiating the two; `BP_predictions_hyperparams-<split>.csv`, files containing the structures in each of train/val/test and the models' predictions on these structures; `hp_sweep.py`, a script used to train the models and generate all of the other files in the subdirectory; `<functional>_best_model.txt`, containing the MAE (in HFX%) and R$^2$ value for the best performing model for that functional; and `<functional>_optimal_hps.txt`, containing the hyperparameters corresponding to the best performing model for that functional.
- The `racs_replicate_dfbp_set` folder:
  - Contains `<functional>-<dataset>_racs_predictions.csv` files, containing the predictions of RACs-based ML models when predicting on functional when using the same train/val split as DF-BP models, as well as `<functional>_racs_replicate.py` scripts, which train RF models using the same splits as PBE0-PBEx or SCAN0-SCANx.
- `RACs Assessment.ipynb`: evaluates the error in RACs-based ML models.