# E3DSECS - Tool to perform volumetric reconstruction of electric current density
This package is intended to explore the feasibility of using EISCAT 3D to do volumetric reconstruction of the 3D electric current density vector field. To assess the feasibility, E3DSECS uses the e3doubt package (https://github.com/Dartspacephysiker/e3doubt) to add realistic noise to a scenario extracted from a model run of the GEMINI model of a simple pair of FACs. The core elements of E3DSECS is the framework to fit a 3D vecror field of current density with a built-in current continuity constrain. The input to E3DSECS is observations of the electric current density perpendicular to the magnetic field, which can be estimated from EISCAT 3D measurements, and possibly also other data.    

## Install instructions
E3DSECS largely builds on existing packages from our group. Most notably Lompe, which also include a number of core dependencies, e.g. secsy, dipole, polplot, ppigrf. Also, E3DOUBT is a key feauture we use here, also having a number of subdependencies. Here is listed a setup tested on a virtual ubuntu box (nrec) that seems to be running fine. This assumes the mamba package manager. Replacing mamba with conda is expected to work just fine.

### Prerequisits: rust and R
#### On Linux
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh  
sudo apt install r-base-dev  

#### On Mac with Homebrew
brew install rust  
brew install r  

### Setting up dediated environment
mamba create -n e3dsecs python=3.12 fortran-compiler pip ipython imageio jupyter cmake  
mamba activate e3dsecs  

### Python packages
pip install great-circle-calculator  
pip install "lompe[deps-from-github,extras] @ git+https://github.com/klaundal/lompe.git@main"  
git clone https://github.com/gemini3d/pygemini  
pip install -e pygemini  
git clone https://github.com/Dartspacephysiker/e3doubt  
pip install --editable ./e3doubt  
cd e3doubt/external  
git clone https://github.com/ilkkavir/ISgeometry.git  
cd ..  
cd ..  

#### E3DOUBT must be initiated before use, should be done at this point, see readme file in e3doubt https://github.com/Dartspacephysiker/e3doubt  


### Clone this package, E3DSECS
git clone https://github.com/jpreistad/e3dsecs.git  
mkdir e3secs/paper/inversion_coefs  

### Download sample datafiles (optional)
To use the example notebooks that produce the figures in the E3DSECS paper (to be submitted), a sample datafile with a corresponding grid file must be downloaded. For the paper we have published output from a GEMINI simulation that is used for benchmarching of the reconstruction.  
The simulation output and grid can be downloaded from: https://doi.org/10.5281/zenodo.10561479  

### Update paths
To run the example notebooks in the ./paper/ folder producing the figures in the paper describing the tehnique, the path to the datafiles must be updated, set at top of each notebook.
