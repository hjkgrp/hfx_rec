The files contained in this directory are:

- `Generate RACs.ipynb`, `csd_racs.csv`, and `vss_racs.csv`:
  - The notebook describes how to generate the RACs used from the xyz files in the DFA recommender repo. The `.csv` files contain the generated RACs features.
- `DF Featurization.ipynb` and the `density_fitting` subdirectory:
  - The notebook describes how to generate features for the DF-BP models from completed psi4 calculations. The `density_fitting` subdirectory contains an example of what the folders used in the generation of the features looked like.
- `BP_features` subdirectory:
  - Contains all of the files needed to train the DF-BP models with various input densities. The first word denotes the functional used to generate the density features. For every dataset, for both CSD-76-HFX and VSS-452-HFX, there is a `X.pkl` file, containing the features, and a `structures.csv` file mapping these back to their identifiers. There is also a `standard_dict.pkl` file, which is used for normalizing the features.