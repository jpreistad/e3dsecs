# E3DSECS - Tool to perform volumetric reconstruction of electric current density


## Install instructions
E3DSECS largely builds on existing packages from our group. Most notably Lompe, which also include a number of core dependencies, e.g. secsy, dipole, polplot, ppigrf. Also, E3DOUBT is a key feauture we use here, also having a number of subdependencies. Here is listed a setup tested on a virtual ubuntu box (nrec) that seems to be running fine. This assumes the mamba package manager. Replacing mamba with conda is expected to work just fine.

### Prerequisits: rust and R
#### On Linux
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
sudo apt install r-base-dev

#### On Mac
brew install rust
brew install r

### Setting up dediated environment
mamba create -n e3dsecs
mamba activate e3dsecs
### Python packages
mamba install pip
pip install great-circle-calculator
pip install "lompe[deps-from-github,extras] @ git+https://github.com/klaundal/lompe.git@main"
mamba install ipython
mamba install imageio
git clone https://github.com/gemini3d/pygemini
pip install -e pygemini
git clone https://github.com/Dartspacephysiker/e3doubt
pip install --editable ./e3doubt
cd e3doubt/external
git clone https://github.com/ilkkavir/ISgeometry.git
cd ..
cd ..
#### E3DOUBT must be initiated before use, should be done at this point

### Finally, clone this package, E3DSECS
git clone https://github.com/jpreistad/e3dsecs.git
mkdir e3secs/paper/inversion_coefs