import sys 
import numpy as np
from tools.kolm_to_channel import BaseFlow, NumericSolver, TimeStepper
from tools.misc_tools import get_ic_file, log_all_params, on_local_device

material_params = {'W': 50,
                   'beta': 0.9,
                   'Re': 0.5,
                   'L': np.infty,
                   'eps': 1e-3,
                   'rho': 0}

system_params = {'ndim': 2,
                 'Lx': 8 * np.pi,
                 'n': 1}

solver_params = {'Nx': 256,
                 'Ny': 256,
                 'dt': 1e-2}

if len(sys.argv) == 3:
    job_idx = int(sys.argv[1])
    rho = float(sys.argv[2])
    material_params['rho'] = rho
elif on_local_device():
    pass
else:
    raise Exception('Need more inputs!')

log_all_params(material_params, system_params, solver_params)

timestepper = TimeStepper(material_params=material_params, system_params=system_params, solver_params=solver_params)

ic_file, noise_coeff = get_ic_file(material_params, system_params, solver_params, suffix=f'recent-', subdir='arrowhead_2D', 
                                   ic_dict_if_reinit={'rho': 0, suffix': 'recent-symmetric'})

timestepper.ic(ic_file=ic_file, flow=None, noise_coeff=1e-3)

timestepper.simulate(T=5000, ifreq=200, 
                     track_TW=False, 
                     enforce_symmetry=False,
                     save_over_long=False, 
                     save_full_data=False, full_save_freq=5,
                     save_subdir='arrowhead_2D', suffix_end='', 
                     plot=True, plot_dev=True, plot_subdirectory='arrowhead_2D')