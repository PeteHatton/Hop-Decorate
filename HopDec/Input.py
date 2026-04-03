import sys

import numpy as np
import xml.etree.ElementTree as ET

class InputParams:
    def __init__(self):

        # =====================================================================
        # Main parameters
        self.runTime = 0
        self.inputDirectory = ""
        self.inputFilename = ""
        self.verbose = 1
        self.maxModelDepth = 0
        self.canonicalLabelling = 0
        self.checkpointInterval = 0
        self.redecorateTransitions = 0
        self.modelSearch = 0

        # =====================================================================
        # Parameters for LAMMPS
        self.LAMMPSInitScript = ""
        self.MDTimestep = 0
        self.NSpecies = 0
        self.specieNames = None
        self.specieNamesString = ''
        self.MDTimestep = 0.0
        self.MDTemperature = 0
        self.segmentLength = 0
        self.atomStyle = 'atomic'

        # =====================================================================
        # Parameters for defect identification
        self.centroN = 0
        self.centroCutoff = 0.0
        self.eventDisplacement = 0
        self.bondCutoff = 0.0
        self.nDefectsMax = 1
        self.maxDefectAtoms = -1

        # =====================================================================
        # Parameters for Redecoration
        self.staticSpeciesString = ''
        self.staticSpecies = None
        self.staticSpeciesTypes = []
        self.activeSpeciesString = ''
        self.activeSpecies = None
        self.activeSpeciesTypes = []
        self.concentrationString = ''
        self.concentration = None
        self.nDecorations = 0
        self.randomSeed = 1234

        # =====================================================================
        # Parameters for LAMMPS Minimization
        self.minimizationForceTolerance = 0.0
        self.minimizationEnergyTolerance = 0.0
        self.minimizationMaxSteps = 0
        self.maxMoveMin = 1

        # =====================================================================
        # Parameters for NEB
        self.breakSym = 0
        self.NEBNodes = 0
        self.NEBClimbingImage = 0
        self.NEBSpringConstant = 0.0
        self.NEBForceTolerance = 0.0
        self.NEBTimestep = 0.0
        self.NEBMaxIterations = 0
        self.NEBmaxBarrier = np.inf
        self.maxNEBsToDo = np.inf
        self.intMinBuffer = 0.0
        self.minBarrier = 0.0

        # =====================================================================
        # Parameters for DIMER
        self.DIMERForceTol = 0.0
        self.DIMERMaxSteps = 0
        self.initialDIMERDisplacementDistance = 0

def getParams(inputParamFile="HopDec-config.xml"):
    """
    Parses an XML file containing input parameters and creates an InputParams object.

    This function reads the specified XML file containing input parameters and creates an instance
    of the `InputParams` class, populating it with the parsed data. The XML file must have a valid
    structure with tags corresponding to the attributes of the `InputParams` class.

    Parameters:
        inputParamFile (str, optional): The name of the XML file to be parsed. The default value is
                                       "HopDec-config.xml".

    Returns:
        InputParams: An object of the InputParams class containing the parsed input parameters.

    Raises:
        ValueError: If the XML file is not found or there is an error during parsing.
    """

    INPUT_PARAMS_TAG = "InputParams"
    try:
        tree = ET.parse(inputParamFile)
        root = tree.getroot()
    except (ET.ParseError, FileNotFoundError) as e:
        raise ValueError(f"Error parsing XML file '{inputParamFile}': {e}")
    
    def parse_numerical(value):
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value.strip()

    input_params = InputParams()
    for element in root.iter():
        if element.tag == INPUT_PARAMS_TAG:
            continue
        if element.text and element.text.strip():
            setattr(input_params, element.tag, parse_numerical(element.text))

    # split specie names into list
    input_params.specieNames = input_params.specieNamesString.split(',')
    input_params.staticSpecies = input_params.staticSpeciesString.split(',')
    input_params.activeSpecies = input_params.activeSpeciesString.split(',')
    
    # make list of numbered types
    if input_params.staticSpecies[0] == '':
        input_params.staticSpeciesTypes = []
    else:
        input_params.staticSpeciesTypes = [ np.where(np.array(input_params.specieNames) == staticSpecie)[0][0] + 1 
                                            for staticSpecie in input_params.staticSpecies ]
    if input_params.activeSpecies[0] == '':
        input_params.activeSpeciesTypes = []
    else:
        input_params.activeSpeciesTypes = [ np.where(np.array(input_params.specieNames) == activeSpecie)[0][0] + 1 
                                            for activeSpecie in input_params.activeSpecies ]

    # split concentrations into list
    input_params.concentration = [ float(c) for c in str(input_params.concentrationString).split(',') ]

    cSum = np.sum(input_params.concentration)
    if cSum != 1:
        sys.exit(f"{__name__}: ERROR: Concentration does not sum to 1. It sums to {cSum}")
    
    return input_params

if __name__ == "__main__":
    pass