from .Transitions import *
from .State import *
from . import Minimize

import pickle as pkl
from typing import List
import networkx as nx

class Model:
        
    """
    A container and manager for state-transition models of defect dynamics in atomistic systems.

    Attributes
    ----------
    params : InputParams
        Simulation parameters.
    initState : State or None
        The initial atomic state of the system.
    stateList : list of State
        List of known atomic states.
    transitionList : list of Transition
        List of known transitions between states.
    graph : networkx.Graph
        Network graph of states and transitions.
    stateWorkCheck : np.ndarray
        Flags for whether states require additional simulation work.
    """

    def __init__(self, params : InputParams):
        
        """
        Initialize a new Model instance.

        Parameters
        ----------
        params : InputParams
            Parameters controlling model behavior and constraints.
        """
        
        self.params = params
        self.initState = None

        self.graph = None

        self.stateList = [] # Contains the state objects
        self.transitionList = [] # Contains the transition Objects
        self.canLabelList = []
        self.stateWorkCheck = np.array([]) # 0 = do work, 1 = dont do work
        

    def __len__(self):

        """
        Return the number of transitions currently in the model.

        Returns
        -------
        int
            Number of transitions.
        """
            
        return len(self.transitionList)
    
    def loadRedecorations(self):

        """
        Load alloy redecoration data (e.g., barrier statistics) for all transitions in the model.

        Populates the `self.redecorations` list with dataframes or `None` if unavailable.
        """

        # TODO: Add logging
        self.redecorations = []
        for trans in self.transitionList:
            df = trans.loadRedecoration()

            if df.empty:
                self.redecorations.append(None)
            else:
                self.redecorations.append(copy.copy(df))
    
    def buildModelGraph(self):

        """
        Build a NetworkX graph of the model from the current list of states and transitions.
        """

        edges = [ ( trans.initialState.nonCanLabel, trans.finalState.nonCanLabel ) 
                    for trans in self.transitionList ]
        
        nodes = [ state.nonCanLabel for state in self.stateList ]

        self.graph = buildNetwork(nodes, edges)

    def findDepth(self, state):
        """
        Compute the shortest path depth from the initial state to a given state.

        Parameters
        ----------
        state : State
            The target state.

        Returns
        -------
        int or float
            Shortest path length (in transitions) from `initState` to `state`.
            Returns np.inf if unreachable.
        """        

        return shortestPath(self.graph, self.initState.nonCanLabel, state.nonCanLabel)
        
    def update(self, workDistribution = [], states = [], transitions = [], connections = []):

        """
        Update the model by adding new states, transitions, or connection groups.

        Parameters
        ----------
        workDistribution : list of State
            States for which simulation work (e.g., MD) was just performed.
        states : list of State
            Newly discovered states to attempt to add.
        transitions : list of Transition
            Newly discovered transitions to attempt to add.
        connections : list of Connection
            Groups of transitions connecting known and new states.

        Returns
        -------
        int
            1 if at least one new state or transition was added; 0 otherwise.
        """

        def cleanData(data):
            return [ x for x in data if x is not None ]
              
        def updateStates(states):
            foundNew = 0
            for s,state in enumerate(states):
                
                if self.checkUniqueness(state):

                    self.stateList.append(state)
                    self.buildModelGraph()
                    depth = self.findDepth(state)

                    # This is a catch for if we find a state but cant resolve a transition to it.
                    if depth == np.inf:
                      print('WARNING: Found State with no valid Transition. Skipping...')
                      _ = self.stateList.pop(-1)
                      continue

                    if depth <= self.params.maxModelDepth or self.params.maxModelDepth < 0: 
                        state.doWork = 1
                    else:
                        state.doWork = 0

                    state.time = self.params.segmentLength
                    log(__name__,'Added New State to Model')
                    foundNew = 1
                else:
                    log(__name__, 'Previously Seen State.')
            
            return foundNew
        
        def updateTransitions(transitions):

            if self.params.maxDefectAtoms == -1:
                maxDefectAtoms = np.inf
            else:
                maxDefectAtoms = self.params.maxDefectAtoms

            foundNew = 0
            for t, transition in enumerate(transitions):
                # HACK: Need to remove the check for maxDefectAtoms....
                if self.checkUniqueness(transition) and transition.initialState.nDefects <= self.params.nDefectsMax and transition.finalState.nDefects <= self.params.nDefectsMax and len(transition.initialState.defectPositions) // 3 <= maxDefectAtoms and len(transition.finalState.defectPositions) // 3 <= maxDefectAtoms:
                    self.transitionList.append(transition)
                    log(__name__,'Added New Transition to Model')
                    foundNew = 1
                    
                    updateStates([transition.initialState, transition.finalState])

                else:
                    # TODO: - this message isnt appropriate if the defect seperated
                    log(__name__, 'Previously Seen Transition')

            return foundNew
        
        # clean the data. it may have NONEs        
        states = cleanData(states)
        transitions = cleanData(transitions)
        connections = cleanData(connections)

        # update MD time in each state where MD was done.
        for state in workDistribution: state.time += self.params.segmentLength

        # generally not used during 'HopDec-main' functionality
        foundNewTrans = updateTransitions(transitions)
        foundNewState = updateStates(states)

        # When updating the model during 'HopDec-main' 
        # we are usually given a Connection object which is handled below.

        # -1 means no limit so we attempt to add every transition to the model.
        if self.params.maxModelDepth < 0:
            foundNewConn = updateTransitions([ trans for connection in connections for trans in connection.transitions ])
        
        # otherwise we need to check state depths.
        else:

            toAdd = []
            foundNewConn = 0

            for connection in connections:

                self.buildModelGraph()

                for t,trans in enumerate(connection.transitions):

                    if self.findDepth(trans.initialState) <= self.params.maxModelDepth:

                    # if currentDepth <= self.params.maxModelDepth: # if the initial state is one in which we would like to search                

                        # toAdd.append(trans)
                        test = updateTransitions(transitions = [trans])
                        foundNewConn = max(foundNewConn,test)

            # foundNewConn = updateTransitions(transitions = toAdd)

        return max(foundNewState, foundNewConn, foundNewTrans)

    def checkUniqueness(self, obj):

        """
        Check whether a State or Transition is already known in the model.

        Parameters
        ----------
        obj : State or Transition
            Object to check.

        Returns
        -------
        bool
            True if unique (not in the model), False otherwise.
        """
        
        if hasattr(obj, 'NAtoms'):
            targetList = self.stateList
     
        elif hasattr(obj, 'initialState'):
            targetList = self.transitionList

        else:
            sys.exit(TypeError('ERROR: checkUniqueness only accepts State and Transition objects.'))

        if self.params.canonicalLabelling:
            if obj.canLabel in [ target.canLabel for target in targetList ]:
                return 0
            else:
                return 1
        else:
            if obj.nonCanLabel in [ target.nonCanLabel for target in targetList ]:
                return 0
            else:
                return 1
    
    def workDistribution(self, size):
        
        """
        Select a list of states weighted by inverse MD time for additional simulation work.

        Parameters
        ----------
        size : int
            Number of states to return.

        Returns
        -------
        list of State
            Sampled states needing further MD exploration.
        """

        inverseTimes = 1 / np.array([ s.time for s in self.stateList ])
        workArray = np.array([ s.doWork for s in self.stateList ])
        inverseTimes = inverseTimes * workArray

        return np.random.choice(self.stateList, p = inverseTimes  / inverseTimes .sum(), size = size)

def checkpoint(model, filename = 'model-checkpoint_latest.pkl'):
    
    """
    Save the current model state to a pickle file.

    Parameters
    ----------
    model : Model
        The model object to serialize.
    filename : str
        Destination filename.
    """

    with open(filename, 'wb') as f:
        pkl.dump(model, f, protocol=4)
    
def setupModel(params, comm = None) -> Model:
    
    """
    Create and initialize a new Model from an input state file.

    Parameters
    ----------
    params : InputParams
        Model configuration parameters.
    comm : MPI.Comm or None
        MPI communicator for distributed execution (optional).

    Returns
    -------
    Model
        A fully initialized Model object with its initial state minimized and inserted.
    """
    
    model = Model(params)

    if params.verbose: log(__name__, f"Reading state file: {params.inputFilename}",0)
    model.initState = read(params.inputFilename)

    Minimize.main(model.initState, params, comm = comm)
    
    model.update(states = [model.initState])

    return model

if __name__ == '__main__':
    pass