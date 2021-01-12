# RUN WITH SOURCE, not BASH!!!

TORCH_VERSION=1.6.0
CUDA_VERSION=cu102
PYTHON_VERSION=3.7
ENV_NAME=rgg

# create venv
conda create --name ${ENV_NAME} python=${PYTHON_VERSION} -y && conda activate ${ENV_NAME}

# install pytorch
conda install pytorch==${TORCH_VERSION} -c pytorch -y

# install pytorch-geometric
pip install torch-scatter==latest+${CUDA_VERSION} -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}.html
pip install torch-sparse==latest+${CUDA_VERSION} -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}.html
pip install torch-cluster==latest+${CUDA_VERSION} -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}.html
pip install torch-spline-conv==latest+${CUDA_VERSION} -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}.html
pip install torch-geometric

# additional pip packages
pip install networkx numpy scikit-learn scipy pyemd pytorch-lightning tqdm dgl-${CUDA_VERSION}

# instal EdEN (for NSPKD)
pip install git+https://github.com/fabriziocosta/EDeN.git

# compile cpps
mkdir -p bin

g++ -std=c++11 dfscode/dfs_code.cpp -o bin/dfscode -O3
g++ -std=c++11 metrics/orca/orca.cpp -o bin/orca -O3
g++ -std=c++11 metrics/isomorph.cpp -O3 -o bin/subiso -fopenmp -I$HOME/boost/include
g++ -std=c++11 metrics/unique.cpp -O3 -o bin/unique -fopenmp -I$HOME/boost/include