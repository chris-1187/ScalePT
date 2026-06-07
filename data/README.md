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
