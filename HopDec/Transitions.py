import os
import random
import copy

import numpy as np
import pickle as pkl
import pandas as pd

from . import State
from .Plots import *
from .Input import *
from .Vectors import *
from .State import *
from .Utilities import *
from .Constants import boltzmann
from .Graphs import graphLabel
import HopDec.Vectors as vec

class Transition:
    """
    Stores information about a single transition from a given state to another
    """
    def __init__(self, initialState : State, finalState : State):

        self.initialState = initialState
        self.finalState = finalState
        self.saddleState = None
                
        self.forwardBarrier = 0
        self.reverseBarrier = 0
        self.dE = 0
        self.KRA = 0

        self.images = [] # states

        self.canLabel = ''
        self.nonCanLabel = ''

        self.redecoration = None
        self.redecorated = 0

    def maxMoveAtom(self):
        _, index = vec.maxMoveAtom(self.initialState, self.finalState)
        type = self.initialState.type[index]
        return index, type

    def loadRedecoration(self):
        ''' Method to load in the redecoration refered to in self.redecoration '''
        filename = f'{self.redecoration}.pkl'
        if os.path.exists(filename):
            with open(filename,'rb') as f:
                df = pkl.load(f)
            return df
        else:
            print("WARNING: Redecoration for this transition does not exist")
            return pd.DataFrame()

    def calcRate(self, temperature : float, prefactor = 1e13) -> float:
        return prefactor * np.exp( - self.forwardBarrier / (temperature * boltzmann) )
    
    def plot(self, folder : str, filename : str):
        
        """
        Plot the NEB energy pathway.

        Args:
            name (str or bool): The filename for the plotted energy pathway.

        """

        if not os.path.exists(folder):
            os.makedirs(folder)

        f = f'{folder}/NEB_{filename}.png'

        log(__name__,f"Saving NEB Energy Pathway to '{f}'")

        ens = np.array([ image.totalEnergy for image in self.images ])
        
        linePlot(f'{f}',
                 xvals = [ i for i,_ in enumerate(self.images) ],
                 xlabel = 'NEB Node',
                 yvals = ens - np.min(ens),
                 ylabel = 'Relative Energy (eV)',
                 )

    def exportStructure(self, folder : str, subfolder : str) -> None:

        """
        Export the NEB structures to a specified folder.

        Args:
            folder (str): The folder where the NEB structures will be exported.

        """

        folder = str(folder)
        subfolder = str(subfolder)

        if not os.path.exists(folder): 
            os.mkdir(folder)
        if not os.path.exists(f'{folder}/{subfolder}'):
            os.mkdir(f'{folder}/{subfolder}')

        log(__name__,f"Saving NEB Structure to '{folder}/{subfolder}'")

        for i, image in enumerate(self.images):
            image.write(f'{folder}/{subfolder}/{i}.dat')
            

    def printTransitionSummary(self):
        """
        Display the results of the energy barriers and related values.

        This function prints the forward energy barriers, reverse energy barriers, KRA (average of forward and reverse barriers),
        and dE values in electron volts (eV).

        Attributes:
        self (object): The instance of the class.

        Returns:
        None

        """
    
        log(__name__,f'Results:')
        print(f'''\t    Forward Energy Barrier: {self.forwardBarrier} eV, Rate at 1000 K: {self.calcRate(1000, self.forwardBarrier):e} 1/s
            Reverse Energy Barrer: {self.reverseBarrier} eV, Rate at 1000 K: {self.calcRate(1000, self.reverseBarrier):e} 1/s
            KRA: {self.KRA} eV
            dE: {self.dE} eV ''')
        
    def label(self, params):

        """
        Get the hash value for a transition.

        Args:
            params: The input parameters.
            initialState: The initial state for the transition.
            finalState: The final state for the transition.

        Returns:
            str: The hash value for the transition.

        """

        initialState = self.initialState
        finalState = self.finalState

        dummyState = initialState.copy()
        dummyState.NAtoms = int((len(initialState.defectPositions) + len(finalState.defectPositions)) // 3)
        dummyState.defectPositions = np.concatenate((initialState.defectPositions, finalState.defectPositions))
        dummyState.pos = np.concatenate((initialState.defectPositions, finalState.defectPositions))

        defectTypes = np.concatenate((initialState.defectTypes, finalState.defectTypes))
        defectIndices = np.concatenate((initialState.defectIndices, finalState.defectIndices))
        
        graphEdges = findConnectivity(dummyState.pos, params.bondCutoff, dummyState.cellDims)
        
        self.canLabel = graphLabel(graphEdges, types = defectTypes, canonical = 1)
        self.nonCanLabel = graphLabel(graphEdges, indices = defectIndices, canonical = 0)

        # dummyState.write(f'state/{self.canLabel}_{self.nonCanLabel}.dat')

class Connection:

    """ A connection is a collection of transitions which connect two or more states"""
    
    def __init__(self, initialState : State, finalState : State):
        
        self.initialState = initialState
        self.finalState = finalState
        self.saddleState = None
        
        self.transitions = []

    def __len__(self):
        return len(self.transitions)

    def printResults(self):

        """
        Display the results of the energy barriers and related values.

        This function prints the forward energy barriers, reverse energy barriers, KRA (average of forward and reverse barriers),
        and dE values in electron volts (eV).

        Attributes:
        self (object): The instance of the class.

        Returns:
        None

        """
        
        fBarString = rBarString = kraString = dEString = ''

        for i in range(len(self.transitions)):
            if i > 0:
                fBarString +=  ", "
                rBarString +=  ", "
                kraString +=  ", "
                dEString +=  ", "

            fBarString +=  f"{self.transitions[i].forwardBarrier}"
            rBarString += f"{self.transitions[i].reverseBarrier}"
            kraString += f"{self.transitions[i].KRA}"
            dEString += f"{self.transitions[i].dE}"
        
        log(__name__,f'Results:')
        print(f'''\t    Forward Energy Barrier(s): {fBarString} eV
            Reverse Energy Barrer(s): {rBarString} eV
            KRA(s): {kraString} eV
            dE(s): {dEString} eV ''')
        
    def plot(self, folder = '.'):
        for t,trans in enumerate(self.transitions):
            trans.plot(folder, t)
    
    def exportStructures(self, folder = '.'):
        for t,trans in enumerate(self.transitions):
            trans.exportStructure(folder,t)