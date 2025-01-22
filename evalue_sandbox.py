import numpy as np
from tools.misc_tools import log_all_params
from tools.kolm_to_channel import BaseFlow, NumericSolver, TimeStepper

material_params = {'W': 30,
                   'beta': 0.9,
                   'Re': 0,
                   'L': np.infty,
                   'eps': 1e-3}

system_params = {'ndim': 2,
                 'n': 1}

solver_params = {'Ny': 128}

log_all_params(material_params=material_params, solver_params=solver_params, system_params=system_params)

evalue_solve = NumericSolver(system_params=system_params, solver_params=solver_params, save_plots=True)

evalue_solve.instability_over_kx(material_params=material_params, kx_list=[0.25, 0.5, 1], init_targets=[1])