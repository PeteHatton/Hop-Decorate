from .Model import *
from .Input import *
from .Utilities import *
from .State import *
from .NEB import *
from .Plots import *
from .Vectors import *
from . import NEB as neb

import numpy as np
import pandas as pd
import random
import pickle as pkl
import copy

class Redecorate:

    def __init__(self, params: InputParams):

        self.params = params

        self.connections = []
        self.aseConnections = []

        self.filename = ''

    def __len__(self):
        return len(self.aseConnections)

    def buildShuffleLists(self, state):

        state_type = np.array(state.type)

        # Initialize initialTypeList with -1 for types not in staticSpeciesTypes
        initialTypeList = np.where(np.isin(state_type, self.params.staticSpeciesTypes), state_type, -1)
        count = np.sum(initialTypeList == -1)

        # Compute nAtEach array, which will hold the number of active species for each type
        nAtEach = count * np.array(self.params.concentration)

        # Create the shuffle list
        shuffleList = np.array([self.params.activeSpeciesTypes[n] 
                                for n, nAt in enumerate(nAtEach) 
                                for _ in range(int(nAt))
                                ])

        # If the shuffle list is shorter than the count, append more of the first element
        if len(shuffleList) < count:
            shuffleList = np.concatenate([shuffleList, np.repeat(self.params.activeSpeciesTypes[0], count - len(shuffleList))])

        return initialTypeList.tolist(), shuffleList.tolist()

    def run(self, initialState: State, finalState: State, comm = None):

        '''
        TODO: Initialize redecs with pathway from static decoration transition
        '''
        
        rank = 0
        size = 1
        startI = 0
        endI = self.params.nDecorations
        newComm = None

        if comm:
            
            rank = comm.Get_rank()
            size = comm.Get_size()
            nDecTot = self.params.nDecorations
            eachRank = nDecTot // size

            startI = rank * eachRank
            endI = startI + eachRank

            newComm = comm.Split(color = rank, key = rank)

        if rank == 0: log(__name__, f"Starting NEB Redecoration Campaign")

        initialTypeList, shuffleList = self.buildShuffleLists(initialState)

        seed = self.params.randomSeed
        
        for n in range(startI,endI):

            shuffleListCurrent = copy.deepcopy(shuffleList)
            init = initialState.copy()
            fin = finalState.copy()
            decoration = np.asarray(copy.deepcopy(initialTypeList))

            log(__name__, f"rank: {rank}: Redecoration: {n+1}",1)        

            # randomize atom type list
            random.seed(seed * (n + 1))
            random.shuffle(shuffleListCurrent)
            
            # recombine with static types
            mask = decoration == -1
            num_to_replace = np.count_nonzero(mask)

            # Ensure there are enough replacement values
            if num_to_replace > len(shuffleListCurrent):
                raise ValueError(
                    f"Not enough values in shuffleListCurrent: "
                    f"need {num_to_replace}, but got {len(shuffleListCurrent)}."
                )

            # Perform the replacement
            decoration[mask] = shuffleListCurrent[:num_to_replace]

            # apply the atom type list to initial and final states.
            init.type = decoration
            fin.type = decoration
            init.NSpecies = self.params.NSpecies
            fin.NSpecies = self.params.NSpecies

            # run a NEB
            connection = neb.main(init, fin, self.params, comm = newComm)

            # if it was successful update decoration list
            if len(connection):
                
                # trim connections
                for trans in connection.transitions:
                    trans.images = []
                    trans.saddleState = None
                connection.redecorationIndex = n + 1
                self.connections.append(connection)   

        if comm:
            comm.barrier()
            if rank == 0: log('Redecoration',f'Gathering results')
            rowsLocal = self.rowsFromConnections(self.params)
            rowsList = comm.gather(rowsLocal, root=0)
            if rank == 0:
                allRows = [r for sub in rowsList for r in sub]
                self.df = pd.DataFrame(allRows)
        else:
            self.df = pd.DataFrame(self.rowsFromConnections(self.params))

        return 0
    
    def rowsFromConnections(self, params: InputParams):
        rows = []
        for d, decoration in enumerate(self.connections):
            for t, transition in enumerate(decoration.transitions):
                rows.append({
                    'Composition': params.concentrationString,
                    'Decoration': decoration.redecorationIndex,
                    'Transition': t + 1,
                    'Initial State Positions': transition.initialState.pos,
                    'Initial State Types': transition.initialState.type,
                    'Initial State Cell Dimensions': transition.initialState.cellDims,
                    'Initial State Canonical Label': transition.initialState.canLabel,
                    'Initial State non-Canonical Label': transition.initialState.nonCanLabel,
                    'Initial State Energy': transition.initialState.totalEnergy,
                    'Final State Positions': transition.finalState.pos,
                    'Final State Types': transition.finalState.type,
                    'Final State Cell Dimensions': transition.finalState.cellDims,
                    'Final State Canonical Label': transition.finalState.canLabel,
                    'Final State non-Canonical Label': transition.finalState.nonCanLabel,
                    'Final State Energy': transition.finalState.totalEnergy,
                    'Transition Canonical Label': transition.canLabel,
                    'Transition non-Canonical Label': transition.nonCanLabel,
                    'Forward Barrier': transition.forwardBarrier,
                    'Reverse Barrier': transition.reverseBarrier,
                    'KRA': transition.KRA,
                    'dE': transition.dE,
                })
                
        return rows
    
    def toDisk(self, filename='test'):
        with open(filename, 'wb') as f:
            pkl.dump(self.df, f)
            
    def summarize(self):
        """
        Iterates through the connections and transitions within each decoration, printing information about each transition.

        Args:
            self (object): The instance of the class containing the connections and transitions.

        Returns:
            None: This function doesn't return anything; it only prints information about the connections and transitions.
        """

        log(__name__,"Summary:")

        for r,decoration in enumerate(self.connections):
            print(f'\tConnection {r+1}:')

            for t,transition in enumerate(decoration.transitions):

                print(f'\t\tTransition {t+1}:')
                print(f'\t\t\t{transition.forwardBarrier = }')
                print(f'\t\t\t{transition.dE = }')

################################################################################

def commandLineArgs():
    """
    Parse command line arguments

    """
    import argparse

    parser = argparse.ArgumentParser(description="Run the NEB method.")
    parser.add_argument('initialFile', help="The initial state file.")
    parser.add_argument('finalFile', help="The final state file.")
    parser.add_argument('-i', dest="paramFile", default="HopDec-config.xml", help="Parameter input file (default=HopDec-config.xml)")
    parser.add_argument('-p', dest="plotPathway", nargs='?', const=True, default=False, help="Plot the energy profile with an optional filename")
    parser.add_argument('-v', dest="verbose", default=False, action="store_true", help="print NEB log to screen")
    
    return parser.parse_args()

################################################################################

def mainCMD(comm):

    from . import State

    rank = 0
    if comm: rank = comm.Get_rank()

    # get command line arguments
    progargs = commandLineArgs()

    # parameters from the config file
    params = getParams()

    # initial state object
    initialState = State.read(progargs.initialFile)
    finalState = State.read(progargs.finalFile)

    transition = Transition(initialState, finalState)

    filename = main(transition, params , comm = comm)

    if rank == 0: log('Redecoration',f'Done. Redecoration results: {filename}')

def main(obj, params : InputParams, comm = None):
    
    rank = 0
    if comm: rank = comm.Get_rank()

    # Redecorate a Transition
    if isinstance(obj, Transition):

        # instantiate redecoration results
        Red = Redecorate(params)
    
        # run the redecoration method
        Red.run(obj.initialState, obj.finalState, comm = comm)

        # record it
        obj.label(params)
        
        filename = f'./{obj.canLabel}_{obj.nonCanLabel}'
        obj.redecoration = filename
        if rank == 0: Red.toDisk(filename = f'{filename}.pkl')

    else:
        raise TypeError("obj must be an instance of Transition")

    if comm: comm.barrier()

    return filename