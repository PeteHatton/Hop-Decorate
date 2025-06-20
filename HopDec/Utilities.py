from ase.io import read, write

import datetime
from typing import List
from HopDec import __version__

def writeTerminalBlock(message : str) -> None:
    
    """
    Writes a message surrounded by '#' characters in a terminal-like block.

    Parameters:
    - message (str): The message to be displayed within the terminal block.

    Returns:
    - None
    
    """

    length = 100
    messagePad = ' ' + message + ' '
    messageLength = len(messagePad)
    if messageLength % 2 != 0:
        messagePad = messagePad + ' '
    
    for _ in range(length): print('#',end='')
    print('')

    for _ in range((length - messageLength) // 2): print('#',end='')
    print(messagePad,end='')
    for _ in range((length - messageLength) // 2): print('#',end='')
    print('')

    for _ in range(length): print('#',end='')
    print('')

def printConsoleHeader():
    def get_clean_version():
        return __version__.split('+')[0]
    writeTerminalBlock(f'Hop-Decorate v{get_clean_version()}')

def printConsoleFooter():
    writeTerminalBlock(f'Fin.')

def log(caller: str, message: str, indent: int = 0) -> None:

    """
    Log output to the screen with optional indentation.

    The function prints the log message to the screen, including the current timestamp,
    the caller name, and the message. It also supports an optional indentation level
    specified by the 'indent' parameter.

    Parameters:
        caller (str): The name of the caller or log source.
        message (str): The message to be logged.
        indent (int, optional): The number of indentation levels (defaults to 0).

    Returns:
        None
    """
    # Only print if less than the level set in Input module
    now = datetime.datetime.now().strftime("%d/%m/%y, %H:%M:%S")
    ind = "  " * indent

    try:
        sp = caller.split('.')
        if sp[0] == 'HopDec':
            caller = sp[-1]
    except:
        pass

    print(f"[{now}]: {ind}{caller} >> {message}", flush = True)

def writeLAMMPSDataFile(filename : str, NAtoms : int, NSpecies: int, cellDims : List[float], types : List[int], positions : List[float]):
        """
        Write LAMMPS data file.

        This function writes a LAMMPS data file with the provided parameters, which can be used as input for LAMMPS simulations.

        Parameters:
            filename (str): The name of the output file to be created.
            NAtoms (int): The total number of atoms in the system.
            NSpecies (int): The number of unique atom types/species in the system.
            cellDims (list): A 1D list of 9 elements representing the cell dimensions for the simulation box.
                            The elements are ordered as [xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz].
            types (list): A 1D list of length NAtoms containing the type/species of each atom in the system.
                        The type values should range from 1 to NSpecies.
            positions (list): A 1D list of length 3*NAtoms containing the atomic positions in the system.
                            The positions should be ordered as [x1, y1, z1, x2, y2, z2, ..., xN, yN, zN].

        Returns:
            None

        Notes:
            - This function will create a new file or overwrite the existing file with the provided 'filename'.
            - The function writes the header information for the LAMMPS data file, including atom count, cell dimensions,
            and atomic positions.

        """
        
        f = open(filename,'w+')
        f.write('# \n')
        f.write(f'{NAtoms} atoms \n')
        f.write(f'{NSpecies} atom types \n')
        f.write(f'0.0 {cellDims[0]} xlo xhi \n')
        f.write(f'0.0 {cellDims[4]} ylo yhi \n')
        f.write(f'0.0 {cellDims[8]} zlo zhi \n')
        f.write(f'\n')
        f.write(f'Atoms # atomic\n')
        f.write(f'\n')
        for i in range(NAtoms):
            f.write(f'{i+1} {types[i]} {positions[3*i]} {positions[3*i+1]} {positions[3*i+2]} \n')
        f.close()
