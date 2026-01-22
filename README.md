# MultistageOT: Multistage optimal transport infers trajectories from a snapshot of single-cell data 

## Overview
Single-cell RNA-sequencing captures a temporal slice, or a snapshot, of a cell differentiation process. A major bioinformatical challenge is the inference of differentiation trajectories from a single snapshot. We present MultistageOT: a Multistage Optimal Transport-based framework for trajectory inference in a snapshot. Application of optimal transport has proven successful for many single-cell tasks, but classical bimarginal optimal transport for trajectory inference fails to model temporal progression in a snapshot. Representing a novel generalization of optimal transport, MultistageOT addresses this major limitation by introducing a temporal dimension, allowing for high resolution modeling of intermediate differentiation stages.

## Contents
This repository contains the MultistageOT package (under `packages/model/multistageot.py`) and scripts for producing results presented in our manuscript "MultistageOT: Multistage optimal transport infers trajectories from a snapshot of single-cell data" (available [here](https://arxiv.org/abs/2502.05241) as an arXiv pre-print). Note: MultistageOT is now published in PNAS and the published article is available online [here](https://www.pnas.org/doi/10.1073/pnas.2516046122)).

This Github repository is intended as a user-friendly, step-by-step documentation to demonstrate the baseline implementations of MultistageOT. For complete reproducibility and code completeness, we host additional scripts and processed data for reproducing all the results on Zenodo (Link: [DOI: 10.5281/zenodo.17233337](https://zenodo.org/records/17233338))

### Table of contents
 * [User guide](#userguide)
   * [Hardware notice](#hardware)
   * [Operating systems](#os)
   * [Installation](#installation)
   * [Example usage](#exampleuse)
   * [Tutorial 1: Overview of MultistageOT](https://github.com/dahlinlab/MultistageOT/blob/main/scripts/notebooks/synthetic_data/main.ipynb)
   * [Tutorial 2: MultistageOT applied to a single cell RNA-sequencing snapshot of hematopoiesis](https://github.com/dahlinlab/MultistageOT/blob/main/scripts/notebooks/real_data/paul2015/main.ipynb)
 * [Scripts for reproducing results](#scripts)
   * [Demo](#demo)
   * [Application on single-cell RNA-sequencing data](#scrnaseq)
 * [License](#license)
  
## User guide <a name="userguide"></a>
### Hardware notice  <a name="hardware"></a>
All the results in [the manuscript](https://arxiv.org/abs/2502.05241) were produced using a 2021 MacBook with Apple M1 Pro and 32 GB memory.

### Operating systems  <a name="os"></a>
MultistageOT is implemented in Python and is thus supported on macOS and Linux. The package has been tested on the following systems:

macOS: Ventura 13.7.3 (Apple M1 Pro) and Sonoma 14.7.4 (Intel Core i7).
	
### Installation <a name="installation"></a>
Running MultistageOT in the notebooks provided in this repository requires Python (we recommend version 3.9 as it was built and run on version 3.9.16) as well as installing the following dependencies (takes seconds):

```
pip install numpy==1.24.3
pip install scanpy==1.9.3
pip install jupyter
```
(Note: We used numpy version 1.24.3 which ensures compatibility with scanpy 1.9.3, but MultistageOT can be run on numpy versions > 2.0).

### Example usage <a name="exampleuse"></a>
Load dependencies:
```
from packages.model.multistageot import MultistageOT
import numpy as np
import pandas as pd
```

We assume the data is represented as a Pandas DataFrame (rows corresponds to cells, and columns corresponds to genes):

```
# One-dimensional example data
X = np.array([[0],
              [1],
              [2],
              [3],
              [4],
              [5]]) 

# Define Pandas DataFrame:
data = pd.DataFrame(X, index=np.arange(6), columns=['gene_expression'])
```

`MultistageOT` is instantiated as a class object. It requires the following input:

* `initial_cells` - a list of indices corresponding to the least mature cells in the data (e.g., stem cells)
* `terminal_cells` - a list of indices corresponding to the most mature cells in the data (e.g., lineage-committed cells).
* `n_groups` - the number of intermediate transport stages, modelling the maximum number of steps required for the cell states in `initial_cells` to differentiate into the cell states in  `terminal_cells`
* `epsilon` - regularization parameter, controlling the level of diffusion in the inferred cell-cell couplings.

(please see the manuscript's Methods-section and Supplementary Note for additional implementational details). 

```
# Instantiate MultistageOT:
msot = MultistageOT(initial_cells  = [0],
                    terminal_cells = [5],
                    n_groups = 4,
                    epsilon  = 0.01)
```

To run MultistageOT on data set, we run the `.fit()` method on the data:

```
# Fit MultistageOT model to data:
msot.fit(data)
```
To infer pseudotime, we call the `.pseudotemporal_order()` method:

```
# Infer pseudotemporal order of cells:
pt = msot.pseudotemporal_order()
```
On this toy data, this yields:
```
pt
0    0.0
1    0.2
2    0.4
3    0.6
4    0.8
5    1.0
Name: pseudotime, dtype: float64
```
To infer cell fate probabilities, we call the `.cell_fate_probabilities(fate_groups)` method, where `fate_groups` is a dictionary of key:value pairs of the form `fate_label` : `index_array`, where `fate_label` is a name (string) of a terminal fate (i.e., 'erythroid'), and  `index_array` is the indices corresponding to that class of cells. In this toy example, we only have a single terminal cell state (cell index 5):

```
fate_groups = { 'Fate 1' :  [5] }

# Infer cell fate probabilities:
cfp = msot.cell_fate_probabilities(fate_groups)
```
Consequently, all cells will be inferred to have probability 1 of ending up in this fate:
```
cfp
	Fate 1
0	1.0
1	1.0
2	1.0
3	1.0
4	1.0
5	1.0
```
## Scripts for reproducing results <a name="scripts"></a>
### Demo <a name="demo"></a>
Under `scripts/notebooks/synthetic_data` we provide a Jupyter notebook script (`main.ipynb`) to demonstrate MultistageOT on a synthetic data set. This can be run within minutes and reproduces all the results on synthetic data presented in the manuscript. 

### Application to single-cell RNA-sequencing data <a name="scrnaseq"></a>
To test the framework on a real single-cell RNA-sequencing data set, we provide another Jupyter notebook  `scripts/notebooks/real_data/paul2015/main.ipynb` which can be run to reproduce all the results on the [Paul et al., (2015)](https://doi.org/10.1016/j.cell.2015.11.013) data set.

Note: To run MultistageOT on the other data sets featured in the manuscript requires downloading publicly available data from [Weinreb et al. (2020)](https://www.science.org/doi/10.1126/science.aaw3381) and [Dahlin et al. (2018)](https://ashpublications.org/blood/article/131/21/e1/37145/A-single-cell-hematopoietic-landscape-resolves-8)

## License [![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause) <a name="license"></a>
This project is licensed under the BSD-3-Clause license.
