from . import Atoms
from . import State
from .Input import *

import ase
from mpi4py import MPI

class ASE:

    """
    Parent class for ASE-based simulations using LAMMPSlib, 
    supporting defect migration analysis with NEB or Dimer methods.

    Parameters
    ----------
    params : InputParams
        Simulation parameters, including species names, LAMMPS commands, etc.
    comm : MPI communicator, optional
        MPI communicator for parallel environments (defaults to None).
    """

    def __init__(self, params : InputParams, comm = None):
        
        self.params = params
        self.comm = comm

        self.atomicNumberDict = {}
        self.atom_type_dict = {}
        for i in range(self.params.NSpecies):
            self.atom_type_dict[self.params.specieNames[i]] = i + 1
            self.atomicNumberDict[i+1] = Atoms.atomicNumber(self.params.specieNames[i])

        self.calculator = lambda x: ase.calculators.lammpslib.LAMMPSlib(
                        lammps_header = ['units metal',
                                        f'atom_style {self.params.atomStyle}',
                                        'atom_modify map array sort 0 0'],
                        lmpcmds=(self.params.LAMMPSInitScript).split("\n"),
                        atom_types = self.atom_type_dict, 
                        keep_alive = True, 
                        # comm = self.comm
                        )
        
    def toState(self, aseAtoms : Atoms):

        """
        Converts an ASE Atoms object into an internal State object for simulation.

        Parameters
        ----------
        aseAtoms : ase.Atoms
            ASE atomic configuration to be converted.

        Returns
        -------
        State
            A State object with atomic positions, types, and cell dimensions.
        """

        pos = aseAtoms.positions
        NAtoms = len(pos)
        
        # initialize state object with NAtoms
        state = State.State(NAtoms)

        # add atomic positions to State and cell dimensions
        state.pos = pos.flatten()
        state.cellDims = [ aseAtoms.cell[0][0], 0, 0, 0, aseAtoms.cell[1][1], 0, 0, 0, aseAtoms.cell[2][2] ]

        # type assignment to State object format.
        typeSymbols = self.params.specieNames
        types = []
        for i in range(NAtoms):
            types.append(typeSymbols.index(aseAtoms.symbols[i]) + 1)
        state.type = types
        state.NSpecies = len(typeSymbols)

        return state

    
