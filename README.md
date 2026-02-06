# ScalePT

```text
ScalePT/ 
├── config/                                 # Configuration files
│   └── cluster_config.yaml                 # Infrastructure setup   
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
│   ├── experiments/                        # Raw metrics
│   └── scalept_demo.ipynb                  # ScalePT Demo Notebook
├── scalept/                                # Main ScalePT package
│   └── infrastructure/
│       └── orchestrator.py                 # Pipeline functions
├── spt-worker/                             # Point Transformer V3 module for worker nodes
│   ├── spt_worker/                         
│   │   ├── serialization                   # Modules for 3D points -> 1D sequence conversion
│   │   │   ├── __init__.py                 
│   │   │   ├── default.py                  
│   │   │   ├── hilbert.py                  # Point serialization using Hilbert curve
│   │   │   └── z_order.py                  # Point serialization using Z-order
│   │   ├── __init__.py  
│   │   ├── dataset.py                      # PyTorch Dataset
│   │   ├── eval.py                         # Evaluation script
│   │   ├── model.py                        # Point Transformer V3
│   │   └── train.py                        # ScalePT-Worker Training Script
│   │── .gitignore
│   │── environment.yml                     # ScalePT-Worker conda env
│   │── environment-cpu.yml                 # ScalePT-Worker conda env (CPU only)
│   │── LICENSE
│   └── README.md
├── .gitignore    
├── .gitmodules
├── environment.yml                         # ScalePT conda env 
├── LICENSE         
├── README.md         
├── train.sh                                # coordinates training on the hosts
└── transfer.py                             # coordinates training on the hosts
```

python -m spt_worker.train \
    --data_path /mnt/nfs_share/kitti/dataset \
    --labels_path /mnt/nfs_share/kitti/dataset \
    --output_dir /mnt/nfs_share/kitti/weights \
    --sequences 04 \
    --batch_size 1 \
    --accumulation_steps 4 \
    --num_workers 4


EVAL:

python -m spt_worker.eval \
    --data_path /mnt/nfs_share/kitti/dataset \
    --labels_path /mnt/nfs_share/kitti/dataset \
    --checkpoint_path /mnt/nfs_share/kitti/weights/final_model.pt \
    --sequences 04


## Setup
```
# Create environment
conda env create -f environment.yml

# Activate environment
conda activate scale-pt

# Install the project
pip install -e .
```




# Remove any old install (safe even if not installed)
pip uninstall -y flash-attn

# Clear pip cache (ensures no prebuilt wheels are used)
pip cache purge

# Clone the FlashAttention repository
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention

# Checkout a stable tag that works with PyTorch 2.1.0 + CUDA 11.8
git checkout v2.5.8

# Build and install from source
pip install . --no-build-isolation


# Verify:
python -c "import torch; import flash_attn; print('torch:', torch.version.cuda); print('flash_attn ok')"


export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH