<img src="logo.png" alt="HopDec Logo" width="300"/>
<br/><br/>

# Hop-Decorate (HopDec)

A high-throughput molecular dynamics workflow for generating atomistic databases of defect transport in chemically complex materials.

---

## Table of Contents

- [Description](#description)
- [Installation](#installation)
  - [Requirements](#requirements)
- [Usage & Functionality](#usage--functionality)
- [Known Limitations](#known-limitations)
- [Bug Reports & Community](#bug-reports--community)
- [License & Copyright](#License)

---

## Description

**Hop-Decorate (HopDec)** is a Python-based automation framework for generating and analyzing defect transport processes in disordered or chemically complex materials (CCMs). It enables high-throughput exploration of migration pathways and kinetic barriers by integrating:

- Molecular dynamics and saddle point search methods (e.g., Dimer)
- Nudged Elastic Band (NEB) energy barrier calculations
- Automated defect transition discovery
- Local chemical redecoration for distribution-based kinetic sampling

HopDec is particularly useful for generating data-driven surrogate models or inputs for kinetic Monte Carlo simulations in systems where chemical disorder (e.g., alloys or doped oxides) plays a significant role in defect dynamics.

---

## Usage & Functionality

HopDec provides two core capabilities:

### 1. **Transition Discovery**
Given an atomic configuration containing a defect:

- Runs molecular dynamics at user-defined temperature **or**
- Performs a Dimer saddle-point search to discover transitions
- Calculates energy barriers via NEB
- Builds a database of defect transitions with associated kinetic data

### 2. **Chemical Redecoration**
For a known transition (either user-specified or discovered):

- Substitutes atomic species according to a specified composition (e.g., 50:50 Cu:Ni)
- Recalculates energy barriers across multiple random decorations
- Returns distributions of barriers and associated statistics

These functionalities combine to produce migration graphs where nodes are configurations and edges are statistically sampled transitions — ideal for downstream kMC or machine-learned modeling.

---

## Installation

### Requirements

HopDec depends on the following external tools and Python libraries:

- [LAMMPS](https://www.lammps.org)
- [openMPI](https://www.open-mpi.org)
- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)
- [matplotlib](https://matplotlib.org/)
- [networkx](https://networkx.org/)
- [ASE](https://wiki.fysik.dtu.dk/ase/)
- [pandas](https://pandas.pydata.org/)
- [mpi4py](https://mpi4py.readthedocs.io/)

We recommend using [Conda](https://docs.conda.io/en/latest/) to manage your environment

Some users have reported issues with installing ASE with conda, if you also have problems consider using pip:
```bash   
pip install ase
```

It is then recommended to add these lines to your .zshrc or .bashrc:  
```bash  
export PATH=$PATH:/path/to/Hop-Decorate/  
export PYTHONPATH=$PYTHONPATH:/path/to/Hop-Decorate/  
```

## License

This program underwent formal release process with Los Alamos National Lab 
with reference number O4739

© 2024. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

This program is Open-Source under the BSD-3 License.
 
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


