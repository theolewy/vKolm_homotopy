import copy
import os
import pickle

import h5py
import numpy as np
import logging
import time
from cfd_tools.cartesian_systems.misc_tools import *
from dedalus.tools import post
import socket


logger = logging.getLogger(__name__)

def get_roots():
    # WHEN RUNNING IN MATHS SERVERS
    projects_path_local = os.path.expanduser('~') + '/Documents/projects/'
    data_path_fawcett = os.path.expanduser('~') + '/../../nfs/st01/hpc-fluids-rrk26/tal43/'

    if os.path.exists(projects_path_local):
        core_root = projects_path_local + 'vKolm_homotopy/'
        data_root = projects_path_local + 'vKolm_homotopy/storage/'
    elif os.path.exists(data_path_fawcett):
        core_root = os.path.expanduser('~') + '/projects/vKolm_homotopy/'
        data_root = data_path_fawcett + 'vKolm_homotopy/'
    else:
        core_root = os.path.expanduser('~') + '/projects/vKolm_homotopy/'

    return core_root, data_root

def get_metric_from_params(material_params, system_params, solver_params, suffix, subdir, metric='trace', deviation=True):

    fpath = get_fpath_sim(material_params, system_params, solver_params, suffix=suffix, subdir=subdir)
    t_all, metric_all = get_metric_from_fpath(fpath, metric=metric, deviation=deviation)

    return t_all, metric_all

# def W_list_from_stored_simulations(material_params, system_params, solver_params, suffix='', subdir=''):

#     _, data_root = get_roots()
#     dir_path = os.path.join(data_root, 'simulations', subdir)
#     W_list = []

#     if not os.path.exists(dir_path): return np.array(W_list)

#     stored_files = os.listdir(dir_path)

#     for fname in stored_files:
#         material_params_2, system_params_2, solver_params_2, fname_suffix_2 = fname_to_params(fname)
#         if material_params_2 is None: continue
#         if params_same(material_params, system_params, solver_params, suffix,
#                 material_params_2, system_params_2, solver_params_2, fname_suffix_2):
#             W_list.append(material_params_2['W'])

#     return np.array(sorted(W_list))

def get_ic_file(material_params, system_params, solver_params, restart=False, suffix='', subdir='', ic_dict_if_reinit=None, **kwargs):

    if 'suffix_name' in kwargs.keys(): suffix = f"recent-{kwargs['suffix_name']}"

    # if closest_made_to_params:

    #     W_list = W_list_from_stored_simulations(material_params, system_params, solver_params, suffix=suffix,
    #                                             subdir=subdir)

    #     material_params = copy.deepcopy(material_params)
    #     W = material_params['W']

    #     if closest_made_to_params == 'lower':
    #         if len(W_list[W_list <= W]) != 0:
    #             W_replacement = np.max(W_list[W_list <= W])
    #             material_params['W'] = W_replacement
    #     elif closest_made_to_params == 'higher':
    #         if len(W_list[W_list >= W]) != 0:
    #             W_replacement = np.min(W_list[W_list >= W])
    #             material_params['W'] = W_replacement
    #     elif closest_made_to_params == 'closest':
    #         if len(W_list) != 0:
    #             W_replacement_idx = np.argmin((W_list - W) ** 2)
    #             W_replacement = W_list[W_replacement_idx]
    #             material_params['W'] = W_replacement
    #     else:
    #         raise Exception("Must be lower or higher")

    save_folder = get_fpath_sim(material_params, system_params, solver_params, subdir=subdir, suffix=suffix, **kwargs)

    if os.path.exists(save_folder) and not restart:
        post.merge_process_files(save_folder, cleanup=True)
        comm = MPI.COMM_WORLD
        comm.Barrier()
        time.sleep(2)   # to ensure merging is done by the time we get s file
        if len(os.listdir(save_folder)) != 0:
            fname = sorted(os.listdir(save_folder), key=lambda x: int(x.split('_s')[-1][:-3]))[-1]
            ic_file = os.path.join(save_folder, fname)
            noise_coeff = 0
        else:
            ic_file = None
            noise_coeff = 1e-2

    else:
        ic_file = None
        noise_coeff = 1e-2

    if ic_file is None and ic_dict_if_reinit is not None:
        ic_file, noise_coeff = get_ic_file(material_params, system_params, solver_params, restart=False, closest_made_to_params=False,
                    suffix=suffix, subdir=subdir, ic_dict_if_reinit=None, **ic_dict_if_reinit)

    return ic_file, noise_coeff


def get_h5_data(material_params, system_params, solver_params, suffix='', subdir='', s=-1):

    fpath = get_fpath_sim(material_params, system_params, solver_params, suffix=suffix, subdir=subdir)

    data_fields, data_metric = get_h5_data_from_fpath(fpath, s)

    return data_fields, data_metric

def get_fpath_sim(material_params, system_params, solver_params, suffix='', subdir='', **kwargs):

    params_copy = copy.deepcopy(material_params)
    params_copy.update(system_params)
    params_copy.update(solver_params)

    for param_name, param in kwargs.items():    # overwrite anything in params with kwargs
        params_copy[param_name] = param

    ndim = params_copy['ndim']

    _, data_root = get_roots()

    Nx, Ny = params_copy['Nx'], params_copy['Ny']
    name = f"sim_W_{params_copy['W']:.6g}_Re_{params_copy['Re']:.6g}_beta_{params_copy['beta']:.6g}_eps_{params_copy['eps']:.6g}_L_{params_copy['L']:.5g}_Lx_{params_copy['Lx']:.5g}_rho_{params_copy['rho']:.6g}_ndim_{ndim}_N_{Nx}-{Ny}_{suffix}/"


    save_folder = os.path.join(data_root, 'simulations', subdir, name.replace('.', ','))

    return save_folder

# def fname_to_params(fname):

#     fname = fname.replace(',', '.')
#     fname_split = fname.split('_')

#     validity_flag = fname_split[0] == 'sim'
#     if not validity_flag: return None, None, None, None
#     fname_W = float(fname_split[2])
#     fname_Re = float(fname_split[4])
#     fname_beta = float(fname_split[6])
#     fname_eps = float(fname_split[8])
#     fname_a = float(fname_split[10])
#     fname_L = float(fname_split[12])
#     fname_Lz = float(fname_split[14])
#     fname_bc = fname_split[16] + '_' + fname_split[17]
#     fname_ndim = int(fname_split[19])
#     N_tuple = tuple([int(N) for N in list(fname_split[21][1:-1].split('-'))])
#     if fname_ndim == 3:
#         fname_Nr, fname_Nth, fname_Nz = N_tuple
#     elif fname_ndim == 2:
#         fname_Nr, fname_Nz = N_tuple
#         fname_Nth = None
#     elif fname_ndim == 1:
#         fname_Nr = N_tuple
#         fname_Nth = fname_Nz = None
#     else:
#         raise Exception

#     fname_suffix = fname_split[22]

#     material_params = {'W': fname_W,
#                        'beta': fname_beta,
#                        'Re': fname_Re,
#                        'L': fname_L,
#                        'a': fname_a,
#                        'eps': fname_eps}

#     system_params = {'bc': fname_bc,
#                      'ndim': fname_ndim,
#                      'Lz': fname_Lz}

#     solver_params = {'Nr': fname_Nr,
#                      'Nth': fname_Nth,
#                      'Nz': fname_Nz}

#     return material_params, system_params, solver_params, fname_suffix


def get_s_list(material_params, system_params, solver_params, suffix='', subdir=''):

    fpath = get_fpath_sim(material_params, system_params, solver_params, suffix=suffix, subdir=subdir)

    get_s_list_from_fpath(fpath)
