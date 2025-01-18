import copy
import logging
import os

import numpy as np
from tools.misc_tools import get_ic_file, get_roots
from dedalus import public as de
from dedalus.tools import post

from mpi4py import MPI
import sys

logger = logging.getLogger(__name__)

_, data_root = get_roots()

simulations_path = os.path.join(data_root, 'simulations')
subdirectories = os.listdir(simulations_path)

for subdirectory in subdirectories:
    subdirectory_path = os.path.join(simulations_path, subdirectory)
    for base_dir in os.listdir(subdirectory_path):
        base_path = os.path.join(subdirectory_path, base_dir)
        post.merge_process_files(base_path, cleanup=True)



logger.info('Done')