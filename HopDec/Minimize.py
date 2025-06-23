from .Lammps import *
from .Input import getParams, InputParams
from .Utilities import log
from .State import read, getStateCanonicalLabel, State

################################################################################

def commandLineArgs():

    """
    Parse command-line arguments for lattice minimization.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments including:
        - inputFile : str
        - outputFile : str
        - dumpMin : bool
    """

    import argparse
    
    parser = argparse.ArgumentParser(description="Minimise a lattice file.")
    
    parser.add_argument('inputFile', help="The lattice file to be minimised.")
    parser.add_argument('outputFile', help="The file to store the minimised lattice in.")
    parser.add_argument('-d', dest="dumpMin", default=False, action="store_true", help="Dump the minimization")
    
    return parser.parse_args()


################################################################################
def mainCMD(comm):

    """
    Entry point for command-line execution to minimize a lattice file.

    This function:
    1. Parses command-line arguments.
    2. Loads minimization parameters.
    3. Reads an atomic structure from a LAMMPS data file.
    4. Minimizes the structure using LAMMPS.
    5. Writes the minimized state to a specified output file.

    Parameters
    ----------
    comm : MPI.Comm or None
        Optional MPI communicator for parallel execution.
    """

    # pull command line arguments
    progargs = commandLineArgs()
    
    # read the minimisation parameters  
    params = getParams()
    
    # read lattice and calculate the forces
    state = read(progargs.inputFile)

    # Minimize
    main(state, params, dump = progargs.dumpMin, verbose = True, comm = comm)
    
    # write relaxed state
    state.write(progargs.outputFile)
    
    log(__name__, f'Minimized state is stored at: {progargs.outputFile}', 2)

def main(state : State, params : InputParams, dump = False, verbose = False, comm = None, lmp = None):

    """
    Perform energy minimization on a `State` object using LAMMPS.

    Parameters
    ----------
    state : State
        Atomic state to minimize.
    params : InputParams
        Simulation parameters, including minimization tolerances.
    dump : bool, optional
        If True, enables LAMMPS dump output (default is False).
    verbose : bool, optional
        If True, prints detailed logging messages (default is False).
    comm : MPI.Comm or None, optional
        MPI communicator for parallel LAMMPS runs (default is None).
    lmp : LammpsInterface or None, optional
        If provided, uses an existing LAMMPS interface (default is None).

    Returns
    -------
    float
        Maximum atomic displacement during minimization.
    """

    # Minimize
    if not lmp: lmp = LammpsInterface(params, communicator = comm)
    move = lmp.minimize(state, dump = dump, verbose = verbose)
    
    # labelling
    getStateCanonicalLabel(state, params, comm = comm, lmp = lmp)

    return move