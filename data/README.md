The files contained in this directory are:

- `CSD-76.csv` and `VSS-452.csv`:
  - Contains the data from the original DFA recommender paper (https://doi.org/10.1038/s43588-022-00384-0). DLPNO-CCSD(T) references used in this work, as well as the HFX-adjusted semilocal functional data for preliminary analysis.
- `raw_csd76_outputs.csv` and `raw_vss452_outputs.csv`:
  - Contains the total energies (using def2-TZVP and HFX-adjusted PBE/SCAN) of both the high-spin and low-spin state for each complex. Entries are left blank if the corresponding calculation did not converge or was spin contaminated. Columns are marked as '\<functional\>' + '\_hfx\_' + '\<amount of hfx\>'. Row names indicate whether the complex is low or high spin.
- `cleaned_csd76_sse.csv` and `cleaned_vss452_sse.csv`:
  - Contains the vertical spin-splitting energies for each TMC. Entries are left blank if the calculation did not converge or was spin contaminated. Columns named analogously to the `raw` files.
- `optimal_tuning.csv`:
  - Contains the data needed for doing optimal tuning calculations (the IP, EA, HOMO, and LUMO for each complex in both the high and low spin state). Columns are marked as follows: '\<amount of HFX\> \<type of data\> \<LS or HS\> \<functional\>'. Type of data refers to IP, EA, HOMO, or LUMO. 
- `tuned_targets.csv` and `energy-tuned_targets.csv`:
  - Contains the tuned HFX for both PBE and SCAN in both the LS and HS states, defined as the single HFX that maximizes agreement between the HOMO-LUMO and fundamental gaps. The file containing the `energy` prefix additionally contains the VSSE when using the tuned HFX.
- `CSD76targets.csv` and `VSS452targets.csv`:
  - Contains the optimal HFX for each structure, defined as the single HFX that maximizes agreement between the DLPNO-CCSD(T) VSSE and the VSSE calculated when using that HFX.
- `csd76_chargeox.csv`:
  - Contains the metal oxidation state and total ligand charge of all structures in CSD-76.
