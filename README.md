# negMIX
This repository contains the author's implementation in PyTorch for the paper "negMIX: Negative Mixup for OOD Generalization in Open-Set Node Classification".

## Environment Requirement

The experiments were conducted on a hardware platform equipped with a NVIDIA RTX 4060Ti GPU and an Intel(R) Core(TM) i7-13700F CPU. The required packages are as follows:

• python == 3.9.18

• torch==2.2.1

• numpy==1.24.1

• scipy==1.12.0

• scikit_learn==1.4.1

## Datasets

pyg_data/ contains the 3 datasets used in our paper, i.e., Cora, CiteSeer, PubMed.

## Code

'GAT_encoder.py' is the multi-layer GAT model.

'main.py' is an example case of the negMIX model for open-set node classification on the Cora dataset.

'data_process.py' is the data processor that contains the dataset split and mask construction.

'utils.py' contains some utility functions.
