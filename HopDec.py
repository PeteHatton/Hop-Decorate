#!/usr/bin/env python

'''
© 2024. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
'''

"""
HopDec: Command-line interface for high-throughput defect migration modeling.

This script allows the user to run different submodules of the HopDec framework, including:
  - Minimize
  - NEB
  - Dimer
  - Redecorate
  - HopDec (full workflow)

It supports MPI parallelism and checks for required Python libraries on launch.
"""

import sys
import importlib

from HopDec.Utilities import log, writeTerminalBlock, printConsoleHeader

################################################################################

availableCommands = {
    "Minimize": "Minimize",
    "NEB": "NEB",
    "Dimer": "Dimer",
    "Input": "Input",
    "Redecorate": "Redecorate",
    "HopDec": "HopDec",
}

commandInfo = {
    "Minimize": "Minimize a state",
    "NEB": "Run the Nudged Elastic Band method",
    "Dimer": "Run the Dimer method",
    "Input": "Testing",
    "Redecorate": "Run a NEB Campaign with Redecoration",
    "HopDec": "The full Hop and Decorate code",
}

requiredLibs = [
    ("numpy", None, None),
    ("matplotlib", None, None),
    ("networkx", None, None),
    ("ase", None, None),
    ("scipy", None, None),
    ("pandas", None, None),
    ("lammps", None, None),
]

################################################################################

def usage():
    
    """
    Return a string describing how to use the CLI.

    Returns
    -------
    str
        Formatted help string showing available submodules.
    """
        
    availCmdString = ""
    for cmd in sorted(availableCommands.keys()):
        if cmd in commandInfo.keys():
            availCmdString += f"{cmd} - {commandInfo[cmd]}\n    "
        else:
            availCmdString += f"{cmd}\n     "

    usageString = f"""
    Usage: HopDec.py NAME_OF_MODULE [OPTIONS] [ARGUMENTS]
         : HopDec.py NAME_OF_MODULE -h will print help"
                  
    NAME_OF_MODULE must be one of:
                      
    {availCmdString}"""
    
    return usageString

def print_version():

    """
    Print the current HopDec version (using versioneer).
    """

    # Import versioneer and get the version number
    import versioneer
    version = versioneer.get_version()
    print("HopDec version:", version)

def checkRequirements():

    """
    Check that all required external libraries are installed and meet version requirements.
    """
    
    errors = []
    for lib, libver, minver in requiredLibs:
        try:
            _module = importlib.import_module(lib)
        
        except ImportError:
            errors.append(f"Could not find required lib: {lib}")
        
        else:
            if libver is not None and minver is not None:
                getver = getattr(_module, libver, None)
                if getver is not None:
                    if callable(getver):
                        installedver = getver()
                    else:
                        installedver = getver

                    def versionTuple(v):
                        tuple(map(int, v.split(".")))
                    ok = versionTuple(installedver) >= versionTuple(minver)
                    
                    if not ok:
                        errors.append(f"Min version not satisfied for lib: {lib} ({installedver} < {minver})")
    
    if len(errors):
        sys.stderr.write("ERROR: requirements not satisfied (details below):\n\n")
        sys.exit("\n".join(errors) + "\n")

################################################################################

def main():
    
    """
    Entry point for the HopDec CLI.

    Launches the appropriate submodule based on command-line input, and
    runs it with MPI awareness. Performs environment checks and graceful startup.
    """
        
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:

        checkRequirements()
        
        if len(sys.argv) == 2 and (sys.argv[1] == "-h" or sys.argv[1] == "--help"):
            print(usage())
            sys.exit(0)

        if len(sys.argv) == 2 and (sys.argv[1] == "-v" or sys.argv[1] == "--version"):
            print_version()
            sys.exit(0)

        if len(sys.argv) < 2 or sys.argv[1] == "-h" or sys.argv[1] not in availableCommands.keys():
            sys.exit(usage())
        
        printConsoleHeader()

    comm.barrier()
    _module = importlib.import_module(f".{availableCommands[sys.argv[1]]}", package="HopDec")

    modname = sys.argv.pop(1)
    sys.argv[0] += f" {modname}"
    if rank == 0:
        log("HopDec", f"Running: {modname}", 0)

    if hasattr(_module, "mainCMD"):
        status = _module.mainCMD(comm)
        pass
    else:
        print(f"ERROR: cannot run: {modname}")
        status = 254

    comm.barrier()
    if rank == 0:
        log("HopDec", f"Finishing: {modname}", 0)
        
        writeTerminalBlock('Fin.')

    comm.barrier()        
    MPI.Finalize()
    sys.exit(0)

################################################################################

if __name__ == '__main__':
    main()


