The notebooks contained in this directory are:

- `Filtering VSSE vs HFX Curves.ipynb`:
  - Filters the datasets to remove any points or structures that do not yield smooth VSSE vs. HFX curves.
- `IP Tuning Targets.ipynb`:
  - Determines the best single HFX as to match the fundamental gap to the HOMO-LUMO gap. Generates data for the tuned approach. Also shows the dataset sizes after filtering out structures with insufficient converged calculations.
- `Visualize HFX Curves.ipynb`:
  - Visualizes the VSSE vs. HFX curves for each structure.
- `Target Generation.ipynb`:
  - Determines the optimal amount of HFX to use with a semilocal functional as to approximate DLPNO-CCSD(T).
- `Visualize Errors.ipynb`:
  - Helps visualize predictions on VSSE vs. HFX curves. Also demonstrates how to get errors in energy units from the HFX predictions.
- `Convert HFX Predictions to Energies.ipynb`:
  - Demonstrates how to convert a predicted HFX% to a VSSE prediction. Also writes files containing model predictions in kcal/mol for all prediction approaches tested.