# SCmapST

## What is SCmapST?

![Figure1_overview.png](https://github.com/LYxiaotai/SCmapST/blob/main/Figure1_overview.png)

Here, we present SCmapST, a deep learning framework that leverages a novel seeded matches strategy to map single cells in SC data to spatial locations in ST data. Building upon a robust batch-correction step, SCmapST identifies reliable seeded matches to accurately guide the information transfer process and mitigate low-quality noise. Then, SCmapST employs the graph attention network (GAT) to iteratively enhance, exchange, aggregate, and propagate information across the SC and ST graphs, enabling the model to capture high-order feature relationships for deriving the SC-to-ST mapping matrix. 


## How to use MMSpa?

### 1. Requirements
  
SCmapST is implemented in the PyTorch framework (tested on Python 3.9.19). We recommend that users run SCmapST on CUDA.

- `Python 3.9`

- Make sure you have [PyTorch](https://pytorch.org/) and [scanpy](https://scanpy.readthedocs.io/en/stable/) installed.

- More details on the dependences can be found in the `environment.txt` file.

- To start using SCmapST, please download the script [Map61.py](https://github.com/LYxiaotai/SCmapST/blob/main/Map61.py)

### 2. Example

The detailed example can be found in `SCmapST_case.py`

The datasets used in the [SCmapST_case.py](https://github.com/LYxiaotai/SCmapST/blob/main/SCmapST_case.py) can be found [here](https://github.com/LYxiaotai/SCmapST/blob/main/Datasets/)
