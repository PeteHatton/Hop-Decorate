import copy

from .Utilities import *
from .Input import *
from .State import *
from .Vectors import *
from .NEB import *

from ase.optimize import BFGS
from ase.constraints import FixAtoms
from ase.dimer import DimerControl, MinModeAtoms, MinModeTranslate
from ase import Atoms

from .ASE import *

class DimerResults:

    """ Results class for the dimer that we can pass out when we have done a dimer """
    
    def __init__(self):
        
        self.foundSaddle = 0
        self.saddleState = None
        self.newMinimaState = None

class Dimer(ASE):
    
    """
    Wrapper for the ASE Dimer method.

    Args:
        params (InputParams): Input parameters for the Dimer.
        communicator (optional): Communication object. Defaults to None.
    """

    def __init__(self, params: InputParams, communicator = None):
        
        super().__init__(params, communicator)

        self.positions = []
        self.initialPositions = []
        self.initState = None
        
    def run(self, state, DVAtoms = []):
        
        """
        Runs the ASE Dimer method.

        Args:
            state (State): Current state.
            DVAtoms (list, optional): List of defect volume atoms. Defaults to [].

        Returns:
            int: 1 if saddle point is found, 0 otherwise.
        """
        
        self.initState = state
        
        p = state.pos
        self.initialPositions = [ [ p[3*i],p[3*i+1],p[3*i+2] ] for i in range(int(len(p)/3)) ]
        
        ase_state = state.toASE(self.params)
        ase_state.pbc = np.array([1,1,1],dtype=bool)
        ase_state.calc = self.calculator(3)    
        
        e0 = ase_state.get_potential_energy()

        mask = np.ones(state.NAtoms)
        mask[DVAtoms-1] = 0
        ase_state.set_constraint(FixAtoms(mask=mask))
        defectCOM = state.defectCOM
        
        # Set up the dimer
        with DimerControl(initial_eigenmode_method='displacement',
                      displacement_method='vector',
                      displacement_center = (float(defectCOM[0]),float(defectCOM[1]), float(defectCOM[2])),
                      displacement_radius = 10,
                      number_of_displacement_atoms = 100, logfile=None
                      ) as d_control:
            d_atoms = MinModeAtoms(ase_state, control = d_control)
            
            # Displace the atoms
            displacement_vector = np.array([ [0.0,0.0,0.0] for i in range(state.NAtoms) ])
            
            if len(DVAtoms):
                for D in DVAtoms:
                    displacement_vector[D-1] = randomVector(3) * self.params.initialDIMERDisplacementDistance # some random normal vector scaled
            else:
                sys.exit("ERROR: Not Currently Implemented. Must Use '-d' flag on Dimer Method")
                

            d_atoms.displace(displacement_vector = displacement_vector)

            # Converge to a saddle point
            with MinModeTranslate(d_atoms, logfile=None) as dim_rlx:
                dim_rlx.run(fmax = self.params.DIMERForceTol, 
                            steps = self.params.DIMERMaxSteps)

            f = ase_state.get_forces()
            e_b = ase_state.get_potential_energy() - e0

            if np.max(np.abs(f)) < self.params.DIMERForceTol and e_b < self.params.NEBmaxBarrier:
                log(__name__, f"Found saddle point, {e_b} eV above inital position")
                self.positions = ase_state.get_positions()
                return 1
            else:
                log(__name__, f"Dimer failed to find saddle.")
                return 0
            
    def nudgeAtomOffSaddle(self):
        
        """Pushes the saddle point into a new minima."""

        log(__name__, f"Pushing saddle point into new minima")
    
        for i in range(self.initState.NAtoms):
            self.positions[i] -= displacement( self.initialPositions[i], self.positions[i], self.initState.cellDims ) *  0.1

def commandLineArgs():
    
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """

    import argparse

    parser = argparse.ArgumentParser(description="Run the Dimer method.")
    parser.add_argument('initialFile', help="The initial state file.")
    parser.add_argument('outputFile', help = 'Output file name')
    parser.add_argument('-i', dest="paramFile", default="HopDec-config.xml", help="Parameter input file (default=HopDec-config.xml)")
    parser.add_argument('-m', dest="minimizeInput", default=False, action="store_true", help="Minimise the initial states first (default=False)")
    parser.add_argument('-d', dest="useDV", default=False, action='store_true', help='Use defect volume to inform initial displacement of atoms')
    parser.add_argument('-p', dest = "pushSaddle", default = False, action= "store_true", help="When a saddle point is found, push it into the new minima")
    parser.add_argument('-l', dest = "logDimer", default = False, action= "store_true", help="Write log to screen")
    # parser.add_argument('-n', dest = "doNEB", default = False, action= "store_true", help="Do a NEB if we find a new state. Only if we push into a new minima")
    
    return parser.parse_args()

def mainCMD(comm):
    
    """
    The main function for the command-line interface.

    Args:
        comm: Communication object.
    """
    
    # parameters from the config file
    params = getParams()

    # get command line arguments
    progargs = commandLineArgs()

    # initial state objects
    initialState = readStateLAMMPSData(progargs.initialFile)

    dimerResults = main(initialState, params, comm, minimizeInput = progargs.minimizeInput, pushSaddle = progargs.pushSaddle, writeFile = progargs.outputFile)

def main(initialState : State, params: InputParams, comm, minimizeInput = False, pushSaddle = True, writeFile = False):
    
    """
    The main function for the Dimer method.

    Args:
        initialState (State): Initial state object.
        params (InputParams): Input parameters for the Dimer.
        comm: Communication object.
        minimizeInput (bool, optional): Whether to minimize the input before running the Dimer. Defaults to False.
        pushSaddle (bool, optional): Whether to push the saddle point. Defaults to True.
        writeFile (bool, optional): Output file name. Defaults to False.

    Returns:
        DimerResults: Results object containing information about the Dimer run.
    """

    # set up results object - this will be returned to user.
    dimerResults = DimerResults()
    
    # minimize the input before doing the dimer.
    lmp = LammpsInterface(params, communicator=comm)
    lmp.minimize(initialState)
    del(lmp)
    
    # need to get atoms that make up the defect volume and pass them to
    getStateCanonicalLabel(initialState, params, comm=comm)
    DVAtoms = initialState.defectIndices
    
    # Dimer object and run
    dimer = Dimer(params, communicator=comm)
    dimerResults.foundSaddle = dimer.run(initialState, DVAtoms)
    
    if dimerResults.foundSaddle:

        # save saddle State into dimerResults
        saddleState = copy.deepcopy(initialState)
        saddleState.pos = dimer.positions.flatten()

        dimerResults.saddleState = saddleState

        if pushSaddle:

            # push structure off of saddle point
            dimer.nudgeAtomOffSaddle()
            minimaState = copy.deepcopy(initialState)
            minimaState.pos = dimer.positions.flatten()

            # minimize
            lmp = LammpsInterface(params, communicator = comm)
            lmp.minimize(minimaState)
            del lmp

            # save new minima structure to dimer results obj
            dimerResults.newMinimaState = minimaState

        if writeFile and dimerResults.foundSaddle and pushSaddle:
            log(__name__, f'Writing file to {writeFile}')
            dimerResults.newMinimaState.writeState(writeFile)
    
    return dimerResults

if __name__ == "__main__":
    pass