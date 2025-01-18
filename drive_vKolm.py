import sys 
import numpy as np
from tools.solvers.kolm_to_channel import BaseFlow, NumericSolver, TimeStepper
from tools.misc_tools import get_ic_file, log_all_params, on_local_device

material_params = {'W': 30,
                   'beta': 0.9,
                   'Re': 0.5,
                   'L': np.infty,
                   'eps': 1e-3,
                   'rho': 1}

system_params = {'ndim': 2,
                 'Lx': 8 * np.pi,
                 'n': 1}

solver_params = {'Nx': 256,
                 'Ny': 256,
                 'dt': 2e-3}

log_all_params(material_params, system_params, solver_params)

timestepper = TimeStepper(material_params=material_params, system_params=system_params, solver_params=solver_params)

ic_file, noise_coeff = get_ic_file(material_params, system_params, solver_params, suffix=f'recent-symmetric', subdir='arrowhead_2D', 
                                   ic_dict_if_reinit=None)

timestepper.ic(ic_file=ic_file, flow=None, noise_coeff=noise_coeff)

timestepper.simulate(T=5000, ifreq=200, 
                     track_TW=False, 
                     enforce_symmetry=True,
                     save_over_long=False, 
                     save_full_data=False, full_save_freq=5,
                     OVERRIDE_LOCAL_SAVE=False,
                     save_subdir='arrowhead_2D', suffix_end='symmetric', 
                     plot=True, plot_dev=True, plot_subdirectory='arrowhead_2D')