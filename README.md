# E3DSECS - Tool to perform volumetric reconstruction of electric current density


# Install instructions
Create new environment
------------------
#Prerequisits for mac
brew install rust
brew install r
#Prerequisits for linux
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
sudo apt install r-base-dev

# Python packages
mamba create -n e3dtest
mamba acticate e3dtest
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
git clone https://github.com/jpreistad/e3dsecs.git
mkdir e3secs/paper/inversion_coefs