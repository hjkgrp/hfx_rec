import pandas as pd
import pickle
import psi4
from pkg_resources import resource_filename, Requirement
import json
import sys
sys.path.append('dfa_recommender')
from dfa_recommender.df_class import DensityFitting
from dfa_recommender.df_utils import get_subtracted_spectra

psi4.set_output_file('output.dat', False)
psi4.set_memory('5000 MB')
psi4.set_options({'BASIS': 'def2-tzvp', 'reference': 'uks', 'SCF__GUESS': 'CORE'})
psi4.core.prepare_options_for_module('SCF')
psi4.core.set_num_threads(8)

xyzfile = 'opt_geo.xyz'
with open('charge-spin-info.json', 'r') as f:
    d = json.load(f)
densfit = DensityFitting(
    wfnpath = 'HS_wfn.npy',
    wfnpath2 = 'LS_wfn.npy',
    xyzfile = xyzfile,
    charge = d['charge'],
    spin = d['spin'],
    basis = 'def2-tzvp')

p_alpha = get_subtracted_spectra(densfit, t='alpha')
p_beta = get_subtracted_spectra(densfit, t='beta')
spec = {'alpha': p_alpha, 'beta': p_beta}
symbol = [densfit.mol.symbol(i).capitalize() for i in range(densfit.mol.natom())]

with open(f'df_output.pkl', 'wb') as fout:
    pickle.dump({'symbol': symbol, 'spec': spec}, fout)