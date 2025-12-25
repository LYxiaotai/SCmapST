# SCmapST

## What is SCmapST?

![Figure1_overview.png](https://github.com/LYxiaotai/SCmapST/blob/main/Figure1_overview.png)

Here, we present SCmapST, a deep learning framework that leverages a novel seeded matches strategy to map single cells in SC data to spatial locations in ST data. Building upon a robust batch-correction step, SCmapST identifies reliable seeded matches to accurately guide the information transfer process and mitigate low-quality noise. Then, SCmapST employs the graph attention network (GAT) to iteratively enhance, exchange, aggregate, and propagate information across the SC and ST graphs, enabling the model to capture high-order feature relationships for deriving the SC-to-ST mapping matrix. 


## How to use MMSpa?

### 1. Requirements
  
SCmapST is implemented in the PyTorch framework (tested on Python 3.9.19). We recommend that users run SCmapST on CUDA.
