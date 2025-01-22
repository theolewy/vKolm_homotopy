import numpy as np
from tools.kolm_to_channel import BaseFlow, NumericSolver, TimeStepper

material_params = {'W': 30,
                   'beta': 0.9,
                   'Re': 0.5,
                   'L': np.infty,
                   'eps': 1e-3}

system_params = {'ndim': 2,
                 'Lx': 4 * np.pi,
                 'n': 1}

solver_params = {'Ny': 512}

base_flow_solve = BaseFlow(system_params=system_params, solver_params=solver_params)

base_flow_solve.ensure_converged_base(material_params=material_params)
base_flow_solve.plot_base_state(fname='test')
