from .Lammps import *
from .State import *    
from .Input import *
from .Utilities import *
from .Vectors import *
from . import Minimize

import copy
import numpy as np

################################################################################
def main(state : State, params : InputParams, dump = None, comm = None, lmp = None, maxMDTime = np.inf, rank = 0, T = None, segmentLength = None, verbose = False):
    
    """
    Perform a molecular dynamics search to identify a defect transition (hop).

    This function runs MD simulations incrementally until a significant atomic 
    displacement is detected (suggesting a transition event). Once detected, the
    configuration is energy-minimized and returned for further analysis.

    Parameters
    ----------
    state : State
        The initial atomic state from which to begin the MD search.
    params : InputParams
        Input parameters object that controls MD settings and thresholds.
    dump : str or None, optional
        Path prefix to enable LAMMPS dump file output (default is None).
    comm : MPI.Comm or None, optional
        Optional MPI communicator for parallel LAMMPS (default is None).
    lmp : LammpsInterface or None, optional
        If provided, uses an existing LAMMPS interface instance (default is None).
    maxMDTime : float, optional
        Maximum cumulative MD time to search for an event (default is ∞).
    rank : int, optional
        MPI rank for logging and identification (default is 0).
    T : float or None, optional
        MD simulation temperature. If None, uses `params.MDTemperature`.
    segmentLength : int or None, optional
        Number of timesteps per MD segment. If None, uses `params.segmentLength`.
    verbose : bool, optional
        If True, enables logging of progress messages.

    Returns
    -------
    lmp : LammpsInterface
        The LAMMPS interface instance used for the run.
    stateTemp : State
        The final (possibly new) atomic state after the MD search.
    flag : int
        Returns 1 if a hop was found and minimized, 0 otherwise.
    """

    time = 0
    flag = 0
    stateTemp = copy.deepcopy(state)
    
    segmentLength = segmentLength or params.segmentLength
    T = T or params.MDTemperature
    lmp = lmp or LammpsInterface(params, communicator=comm)

    while not flag and time < maxMDTime:
        
        if verbose: log(__name__, f'rank {rank}: Running MD in state: {state.canLabel} - {state.nonCanLabel}',1)
        maxMove = lmp.runMD(stateTemp, segmentLength, T = T, dump = dump)
        
        if maxMove > params.eventDisplacement:

            flag = 1
            if verbose: log(__name__, f'rank {rank}: Transition detected in state: {state.canLabel} - {state.nonCanLabel}')
            Minimize.main(stateTemp, params, verbose = False, lmp = lmp)

        time += params.segmentLength
        state.time += params.segmentLength

    return lmp, stateTemp, flag
    
