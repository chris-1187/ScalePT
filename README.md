# ScalePT: A Framework for Distributed Semantic Segmentation of Large-Scale Mobile LiDAR Point Clouds

## Overview

This is ScalePT, a framework for distributed semantic segmentation of large-scale mobile LiDAR point clouds. It includes 
modules for advanced tiling and fusion mechanisms of point clouds, to reduce memory bottlenecks on standard hardware. At 
its core, the Point Transformer V3 (PTv3) implementation of [Pointcept](https://github.com/Pointcept/Pointcept/tree/main) is 
used. ScalePT builds distributed training and inference pipelines around the PTv3 model to allow the benchmarking of our 
proposed tiling and fusion strategies, in a scalable and distributed setup.

## Installation

### Requirements
For deployment, access to a node cluster with NVIDIA GPUs is required. Each node should have at least the following 
specifications:
- 1 NVIDIA GPU of min. Pascal architecture per Node
- 4 GB of VRAM per GPU
- Shared NFS
- Ubuntu 24.04.3 LTS
- CUDA 11.8
- PyTorch 2.1.0
- spconv 2.3.8
- Redis 7.0.15
- conda 25.9.1

### Setup

The cluster configuration is managed in the `config/config.yaml` file. Here, all node and ssh information needs to be 
specified.

The conda environment for the local client can be created from the 'scale-pt' environment.yml file.
```
# Create environment
conda env create -f environment.yml

# Activate environment
conda activate scale-pt

# Install the project
pip install -e .
```

The conda environment for the worker nodes can be created automatically through running the `setup_remote_environments` 
function of the *ClusterOrchestrator*.

The SemanticKITTI dataset can be downloaded from [here](https://www.semantic-kitti.org/dataset.html#download). It should 
be set up in its default structure at the following location:
```text
ScalePT/ 
└── data/                                   
    └── kitti/                              # SemanticKITTI dataset
        └── dataset/                        
            └── sequences/
                ├── 00/                     # Individual sequences/drives
                │   ├── labels/             # Labels for each frame 
                │   │   ├── 000000.label    # uint32 label for each point
                │   │   ├── ... 
                │   │   └── 004540.label
                │   ├── velodyne/           # Point clouds for each frame
                │   │   ├── 000000.bin      # float32 points as [x,y,z,remission]
                │   │   ├── ... 
                │   │   └── 004540.bin
                │   ├── calib.txt           # Calibration file
                │   ├── poses.txt           # Pose matrices for each frame
                │   └── times.txt           # Timestamps for each frame
                ├── ...
                └── 21/ 
```
The dataset can be initially transferred to the cluster nodes by running the `deploy_dataset` function of the 
*ClusterOrchestrator*. The worker node code can also be initially transferred to the cluster nodes by running the 
`deploy_worker_code` function of the *ClusterOrchestrator*.

## Quickstart

The following shows the main steps needed to train and evaluate with ScalePT. Further ready-to-use code can be found in 
the `notebooks/scalept_demo.ipynb` notebook.

### Setup
To set the worker node cluster up for training and inference, the *ClusterOrchestrator* offers a few convenient functions 
that deploy the necessary components. Basic prerequisite is the correct setup of the cluster `config.yaml`.
```python
### SETUP

from scalept.infrastructure.orchestrator import ClusterOrchestrator

# Target Node Count (1, 2, or 3)
NODE_COUNT = 3

# Initialize cluster
cluster = ClusterOrchestrator(num_nodes=NODE_COUNT)

# Sync dataset (if not already on NFS)
cluster.deploy_dataset()

# Sync code to all nodes
cluster.deploy_worker_code()

# Set up environments on all nodes
cluster.setup_remote_environments()
```

### Training
To train a model on the cluster, the *ClusterOrchestrator* offers a function that can be used to start the training pipeline.
```python
### TRAINING

from scalept.infrastructure.orchestrator import ClusterOrchestrator

# Target Node Count (1, 2, or 3)
NODE_COUNT = 3

# Initialize cluster
cluster = ClusterOrchestrator(num_nodes=NODE_COUNT)

# Launch the training
cluster.run_distributed_training(sampling_strategy="hilbert",               # ['block', 'hilbert', 'knn']
                                 sequences="00 01 02 03 04 05 06 07 09 10", # ["00 01 02 03 04 05 06 07 09 10"]
                                 epochs=100,
                                 custom_tag=f"{NODE_COUNT}nodes"
                                 )

# When a long running training is interrupted, the training can be resumed with the following command:
cluster.resume_distributed_training("[eval ID]")
```

### Inference
To run inference and evaluate a trained model on the cluster, the *ClusterOrchestrator* offers a function that can be 
used to start the inference pipeline. Tiling and fusion strategies need to be specified as parameters.
```python
### INFERENCE

from scalept.infrastructure.orchestrator import ClusterOrchestrator

# Target Node Count (1, 2, or 3)
NODE_COUNT = 3

# Initialize cluster
cluster = ClusterOrchestrator(num_nodes=NODE_COUNT)

# Inference and evaluation
cluster.run_evaluation(
    experiment_path_relative="hilbert_100ep_20260403_200500", # training id
    sampling_strategy="hilbert",                              # ['block', 'hilbert', 'fps_knn', 'voxel_knn', 'nuc_knn', 'kdtree_knn']
    fusion_strategy="logit_average",                          # ['logit_average', 'mc_uncertainty']
    sequence="08",                                            # SemanticKITTI validation sequence
    id=f"{NODE_COUNT}nodes",
    total_frames=2000                                         # Number of frames to evaluate
)
```


## Project Structure
```text
ScalePT/ 
├── config/                                 # Configuration files
│   └── config.yaml                         # ScalePT configuration   
├── data/                                   # Datasets
│   └── kitti/                              # SemanticKITTI dataset
│       └── dataset/                        
│           └── sequences/
│               ├── 00/                     # Individual sequences/drives
│               │   ├── labels/             # Labels for each frame 
│               │   │   ├── 000000.label    # uint32 label for each point
│               │   │   ├── ... 
│               │   │   └── 004540.label
│               │   ├── velodyne/           # Point clouds for each frame
│               │   │   ├── 000000.bin      # float32 points as [x,y,z,remission]
│               │   │   ├── ... 
│               │   │   └── 004540.bin
│               │   ├── calib.txt           # Calibration file
│               │   ├── poses.txt           # Pose matrices for each frame
│               │   └── times.txt           # Timestamps for each frame
│               ├── ...
│               └── 21/ 
├── notebooks/                              # Jupyter Notebooks
│   ├── experiments/                        # Raw experiments/metrics/logs
│   │   └── [training-ID]                   # List of training runs
│   │       ├── inference/
│   │       │   └── [inference-ID]          # List of inference runs                
│   │       │       └── predictions/ 
│   │       └── weights/metrics
│   ├── analytics.ipynb                     # Custom analytics notebook
│   └── scalept_demo.ipynb                  # ScalePT Demo Notebook
├── scalept/                                # Main ScalePT package
│   ├── analysis/                           # Analytics modules
│   │   └── analyzer.py                     # Plots etc.
│   └── infrastructure/
│       └── orchestrator.py                 # Main ClusterOrchestrator
├── spt-worker/                             # Submodule for ScalePT-Worker
│   ├── spt_worker/                         # Main worker package
│   │   ├── serialization                   # Modules for 3D points -> 1D sequence conversion
│   │   │   ├── __init__.py                 
│   │   │   ├── default.py                  
│   │   │   ├── hilbert.py                  # Point serialization using Hilbert curve
│   │   │   └── z_order.py                  # Point serialization using Z-order
│   │   ├── __init__.py  
│   │   ├── dataset.py                      # PyTorch Dataset (DataLoader)
│   │   ├── eval.py                         # Inference/Evaluation script
│   │   ├── model.py                        # Point Transformer V3
│   │   └── train.py                        # ScalePT-Worker Training Script
│   │── .gitignore
│   │── environment.yml                     # ScalePT-Worker conda env
│   │── LICENSE
│   └── README.md
├── .gitignore    
├── .gitmodules
├── environment.yml                         # ScalePT conda env 
├── LICENSE         
├── pyproject.toml                          # Python project file         
└── README.md                            
```
