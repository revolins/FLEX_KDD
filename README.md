# A minimally-reproducible example for synthetic OOD split on ogbl-collab - "Subgraph Generation for Generalizing to Out-of-Distribution Links"

Implementation assumes: conda, 24GB GPU RAM

1. Download dataset in Flex_Example directory: https://drive.google.com/file/d/1veCboeyLRk_oCXpat29L8JZC0653PvJk/view?usp=sharing
2. tar xvf dataset.tar.gz -C Flex_Example
3. environment.yml assumes that conda is installed: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
4. conda env create -f environment.yml
5. conda activate py39
6. bash ncn.sh 0 #specify device number with integer