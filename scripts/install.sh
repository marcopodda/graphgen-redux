# RUN WITH SOURCE, not BASH!!!

TORCH_VERSION=1.6.0
CUDA_VERSION=cu102
PYTHON_VERSION=3.7
ENV_NAME=gen

# create venv
conda create --name ${ENV_NAME} python=${PYTHON_VERSION} -y && conda activate ${ENV_NAME}

# install base packages
conda install scikit-learn pandas joblib networkx pyyaml seaborn ipython jupyter libboost -y

# install pytorch
conda install pytorch==${TORCH_VERSION} -c pytorch -y

# install rdkit
conda install rdkit -c rdkit -y

# additional pip packages
pip install pyemd
pip install pytorch-lightning

# instal EdEN (for NSPKD)
pip install git+https://github.com/fabriziocosta/EDeN.git

# compile cpps
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH
mkdir -p bin

g++ -std=c++11 datasets/dfs_code.cpp -o bin/dfscode -O3
g++ -std=c++11 evaluation/metrics/orca/orca.cpp -o bin/orca -O3
g++ -std=c++11 evaluation/metrics/isomorph.cpp -O3 -o bin/subiso -fopenmp -I$CONDA_PREFIX/include
g++ -std=c++11 evaluation/metrics/unique.cpp -O3 -o bin/unique -fopenmp -I$CONDA_PREFIX/include
