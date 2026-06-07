import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.patches as mpatches

from pandas.io.formats.style import Styler

# KITTI class names mapped to their evaluation indices
CLASS_NAMES = [
    "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "bicyclist",
    "motorcyclist", "road", "parking", "sidewalk", "other-ground", "building",
    "fence", "vegetation", "trunk", "terrain", "pole", "traffic-sign"
]

# Official SemanticKITTI color map (mapped to RGB [0, 1] for matplotlib - converted from BGR)
KITTI_COLOR_MAP = {
    0: [0.0, 0.0, 0.0],               # unlabeled
    10: [100/255, 150/255, 245/255],  # car
    11: [100/255, 230/255, 245/255],  # bicycle
    15: [30/255, 60/255, 150/255],    # motorcycle
    18: [80/255, 30/255, 180/255],    # truck
    20: [0.0, 0.0, 255/255],          # other-vehicle
    30: [255/255, 30/255, 30/255],    # person
    31: [255/255, 40/255, 200/255],   # bicyclist
    32: [150/255, 30/255, 90/255],    # motorcyclist
    40: [255/255, 0.0, 255/255],      # road
    44: [255/255, 150/255, 255/255],  # parking
    48: [75/255, 0.0, 75/255],        # sidewalk
    49: [175/255, 0.0, 75/255],       # other-ground
    50: [255/255, 200/255, 0.0],      # building
    51: [255/255, 120/255, 50/255],   # fence
    70: [0.0, 175/255, 0.0],          # vegetation
    71: [135/255, 60/255, 0.0],       # trunk
    72: [80/255, 240/255, 150/255],   # terrain
    80: [150/255, 240/255, 255/255],  # pole
    81: [255/255, 0.0, 0.0],          # traffic-sign
}
# SemanticKITTI learning map (maps raw labels to 0-18 eval classes, moving to static, ignore to -1)
LABEL_MAP = {
    0: -1, 1: -1, 10: 0, 11: 1, 13: 4, 15: 2, 16: 4, 18: 3, 20: 4, 30: 5,
    31: 6, 32: 7, 40: 8, 44: 9, 48: 10, 49: 11, 50: 12, 51: 13, 52: -1,
    60: 8, 70: 14, 71: 15, 72: 16, 80: 17, 81: 18, 99: -1, 252: 0,
    253: 6, 254: 5, 255: 7, 256: 4, 257: 4, 258: 3, 259: 4
}

# Clean RGB color map for the 0-18 indices
TRAIN_COLOR_MAP = {
    0: [100/255, 150/255, 245/255],  # car (dark blue)
    1: [100/255, 230/255, 245/255],  # bicycle (cyan)
    2: [30/255, 60/255, 150/255],    # motorcycle (deep blue)
    3: [80/255, 30/255, 180/255],    # truck (purple)
    4: [200/255, 0.0, 0.0],          # other-vehicle (dark red)
    5: [255/255, 120/255, 30/255],   # person (orange - FIXED CLASH)
    6: [255/255, 40/255, 200/255],   # bicyclist (magenta)
    7: [150/255, 30/255, 90/255],    # motorcyclist (dark magenta)
    8: [255/255, 0.0, 255/255],      # road (pink/magenta)
    9: [255/255, 150/255, 255/255],  # parking (light pink)
    10: [75/255, 0.0, 75/255],       # sidewalk (dark purple/brown)
    11: [175/255, 0.0, 75/255],      # other-ground (maroon)
    12: [255/255, 200/255, 0.0],     # building (yellow/gold)
    13: [255/255, 120/255, 50/255],  # fence (orange/brown)
    14: [0.0, 175/255, 0.0],         # vegetation (green)
    15: [135/255, 60/255, 0.0],      # trunk (brown)
    16: [80/255, 240/255, 150/255],  # terrain (light green)
    17: [150/255, 240/255, 255/255], # pole (light cyan)
    18: [255/255, 0.0, 0.0],         # traffic-sign (pure bright red - FIXED CLASH)
    -1: [0.0, 0.0, 0.0]              # unlabeled
}

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 16,
    'axes.titlesize': 14,
    'xtick.labelsize': 14.5,
    'ytick.labelsize': 14,
    'legend.fontsize': 14
})

# The specific classes that show in the legend
PRESENTATION_CLASSES = [0,1,2,3, 4,5,6,7, 8, 9, 10,11, 12, 13, 14,15, 16,18]

class ExperimentAnalyzer:

    def __init__(self, experiments_root_dir: str):
        self.root_dir = Path(experiments_root_dir)
        self.experiments = {}
        self._load_all_metrics()

    def _load_all_metrics(self):
        """Scans the experiments directory and parses all metrics.jsonl files."""
        if not self.root_dir.exists():
            print(f"Directory {self.root_dir} does not exist.")
            return

        for exp_dir in self.root_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            exp_name = exp_dir.name
            training_logs = []
            evaluation_logs = []

            # training
            metrics_file = exp_dir / "metrics.jsonl"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            if "epoch" in data:
                                training_logs.append(data)
                        except json.JSONDecodeError:
                            continue
            # inference
            inference_dir = exp_dir / "inference"
            if inference_dir.exists() and inference_dir.is_dir():
                for eval_run_dir in inference_dir.iterdir():
                    if eval_run_dir.is_dir():
                        eval_file = eval_run_dir / "evaluation_metrics.json"
                        if eval_file.exists():
                            try:
                                with open(eval_file, 'r') as f:
                                    eval_data = json.load(f)
                                    eval_data["inference_id"] = eval_run_dir.name
                                    evaluation_logs.append(eval_data)
                            except json.JSONDecodeError:
                                continue

            self.experiments[exp_name] = {
                "training": training_logs,
                "evaluations": evaluation_logs
            }

    def get_evaluations(self) -> pd.DataFrame:
        rows = []
        for exp_name, data in self.experiments.items():
            if not data["evaluations"]:
                continue

            parts = exp_name.rsplit('_', 2)
            if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
                model_name = parts[0]
            else:
                model_name = exp_name

            for eval_data in data["evaluations"]:
                params = eval_data.get("parameters", {})
                results = eval_data.get("results", {})

                inf_id = eval_data.get("inference_id", "unknown_eval")

                # clean inference run name
                if inf_id.startswith("eval_"):
                    eval_name = inf_id[5:]
                else:
                    eval_name = inf_id

                # This is custom and must be adjusted
                if (eval_name not in ['hilbert_mc_uncertainty_soft_o2000',
                                      'block_seq08_8_05',
                                      'hilbert_mc_uncertainty_p2000_o5000',
                                      'hilbert_mc_uncertainty_p2000_o2000',
                                      'hilbert_logit_average_p2000_o5000',
                                      'hilbert_logit_average_p2000_o2000']
                        and "40ep" not in model_name
                        and "4071" not in eval_name
                        and results.get("average_cleanup_iterations", np.nan) == 0):
                    rows.append({
                        #"Model": model_name,
                        "Eval": eval_name,
                        "Strategy": params.get("sampling_strategy", "N/A"),
                        "mIoU (%)": results.get("mean_iou", np.nan),
                        "mAcc (%)": results.get("mean_accuracy", np.nan),
                        "OA (%)": results.get("overall_accuracy", np.nan),
                        "Sampling (ms)": results.get("sampling_latency_ms", np.nan),
                        "Model-Lat (ms)": results.get("model_latency_ms", np.nan),
                        "Fusion (ms)": results.get("fusion_latency_ms", np.nan),
                        "E2E-Lat (ms)": results.get("total_e2e_latency_ms", np.nan),
                        "Cluster Time (min)": results.get("total_cluster_time_min", np.nan),
                        "OS-Factor": results.get("average_oversampling_factor", np.nan),
                        "IP %": results.get("average_interpolated_percentage", np.nan),
                        "AVG Tiles": results.get("average_tiles_per_frame", np.nan),
                    })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(by="mIoU (%)", ascending=False).reset_index(drop=True)
        return df

    def get_evaluation(self, experiment_id: str, inference_id: str) -> Styler:
        rows = []

        # check if exists
        if experiment_id not in self.experiments:
            print(f"Experiment {experiment_id} not found.")
            return pd.DataFrame().style

        data = self.experiments[experiment_id]
        if not data["evaluations"]:
            print(f"No evaluations found for {experiment_id}.")
            return pd.DataFrame().style

        target_eval = None
        for eval_data in data["evaluations"]:
            if eval_data.get("inference_id") == inference_id:
                target_eval = eval_data
                break

        if target_eval is None:
            print(f"Inference run '{inference_id}' not found in experiment '{experiment_id}'.")
            return pd.DataFrame().style

        # get metrics
        results = target_eval.get("results", {})
        per_class_iou = results.get("per_class_iou", {})
        per_class_acc = results.get("per_class_acc", {})

        for c in per_class_iou:
            class_idx = int(c)
            class_name = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else f"Unknown ({c})"

            rows.append({
                "Class": class_name,
                "Class IoU (%)": per_class_iou[c],
                "Class Acc (%)": per_class_acc[c],
            })

        df = pd.DataFrame(rows)

        short_exp_id = experiment_id.split('_', 2)[-1] if '_' in experiment_id else experiment_id
        df_cap = df.style.set_caption(f"<h4>Exp: {short_exp_id} <br><br> Eval: {inference_id}</h4><br>")

        return df_cap

    def plot_training_convergence(self, experiment_ids: list = None, figsize=(10, 6)):
        """
        Plots the training loss curves for the selected experiments on the left y-axis,
        and the learning rate schedule on the right y-axis. X-axis represents epochs.
        """
        if not experiment_ids:
            experiment_ids = list(self.experiments.keys())

        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = ax1.twinx()

        cmap = plt.get_cmap("tab10")
        lines = []
        labels = []

        # Plot the learning rate
        first_exp = experiment_ids[0]
        if first_exp in self.experiments and self.experiments[first_exp]["training"]:
            train_data = self.experiments[first_exp]["training"]
            epochs = [entry["epoch"] for entry in train_data]
            lrs = [entry["learning_rate"] for entry in train_data]

            # Plot LR on right axis (ax2)
            l_lr, = ax2.plot(epochs, lrs, color='gray', linestyle='--', linewidth=2, label="Learning Rate")
            lines.append(l_lr)
            labels.append("Learning Rate Schedule")

        # Plot the training losses for all strategies
        for i, exp_id in enumerate(experiment_ids):
            if exp_id not in self.experiments or not self.experiments[exp_id]["training"]:
                continue

            train_data = self.experiments[exp_id]["training"]
            epochs = [entry["epoch"] for entry in train_data]
            losses = [entry["train_loss"] for entry in train_data]

            # Clean display name
            parts = exp_id.rsplit('_', 2)
            if "block" in parts[0]:
                display_name = "2D Planar Block Sampling Strategy"
            elif "hilbert" in parts[0]:
                display_name = "Hilbert Sampling Strategy"
            elif "knn" in parts[0]:
                display_name = "kNN Sampling Strategy"

            # Plot loss on left axis (ax1)
            color = cmap(i % 10)
            l_loss, = ax1.plot(epochs, losses, color=color, linestyle='-', linewidth=2, label=f"{display_name} Loss")
            lines.append(l_loss)
            labels.append(f"{display_name} Loss")

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Cross Entropy Loss")
        ax2.set_ylabel("Learning Rate")

        ax1.grid(True, linestyle='--', alpha=0.6)

        ax1.legend(lines, labels, loc='upper right', framealpha=0.9)

        plt.tight_layout()

        plt.savefig("training_loss.pdf", format="pdf", bbox_inches="tight")
        plt.show()

    def visualize_frame(self, experiment_ids: list, inference_ids: list, sequence: str, frame_id: str,
                        figsize=(28, 10), subsample_rate=1, zoom=2.6, elev=65, azim=-60):
        """
        Visualizes a specific point cloud frame, comparing ground truth vs. two predicted labels
        from different experiment runs.
        Params:
            experiment_ids: List of two experiment directory names (e.g. ['block_100ep', 'hilbert_100ep'])
            inference_ids: List of two evaluation run names (e.g. ['eval_block_seq08', 'eval_hilbert_seq08'])
            sequence: Sequence ID as string (e.g. '08')
            frame_id: Frame number as string (e.g. '000000')
            figsize: Tuple for the matplotlib figure size
            subsample_rate: Integer to plot every Nth point
        """
        if not isinstance(experiment_ids, list) or len(experiment_ids) != 2:
            print("Error: 'experiment_ids' must be a list containing exactly two experiment ID strings.")
            return

        if not isinstance(inference_ids, list) or len(inference_ids) != 2:
            print("Error: 'inference_ids' must be a list containing exactly two inference ID strings.")
            return

        # --- Dynamic Subtitle Parser ---
        def format_inference_title(inf_id: str) -> str:
            if not inf_id.startswith("eval_"):
                return inf_id

            parts = inf_id.split('_')
            if len(parts) < 4:
                return inf_id

            tiling_raw = parts[1]
            tiling_map = {"block": "Block", "hilbert": "Hilbert", "knn": "kNN", "nuc": "NUC", "voxel": "Voxel"}
            tiling = tiling_map.get(tiling_raw, tiling_raw.capitalize())

            fusion_raw = f"{parts[2]}_{parts[3]}"
            if fusion_raw == "logit_average":
                fusion = "Average Logits"
            elif fusion_raw == "mc_uncertainty":
                fusion = "MC Uncertainty"
            else:
                fusion = fusion_raw.replace("_", " ").title()

            return f"Tiling: {tiling}, Fusion: {fusion}"

        formatted_subtitle_1 = format_inference_title(inference_ids[0])
        formatted_subtitle_2 = format_inference_title(inference_ids[1])
        # -------------------------------

        project_root = self.root_dir.resolve().parent.parent
        data_dir = project_root / "data" / "kitti" / "dataset" / "sequences" / sequence

        velodyne_path = data_dir / "velodyne" / f"{frame_id}.bin"
        gt_label_path = data_dir / "labels" / f"{frame_id}.label"

        exp_id_1, exp_id_2 = experiment_ids
        inf_id_1, inf_id_2 = inference_ids

        pred_label_path_1 = self.root_dir / exp_id_1 / "inference" / inf_id_1 / "predictions" / sequence / f"{frame_id}.label"
        pred_label_path_2 = self.root_dir / exp_id_2 / "inference" / inf_id_2 / "predictions" / sequence / f"{frame_id}.label"

        # validations
        if not velodyne_path.exists():
            print(f"Error: Point cloud not found at {velodyne_path}")
            return
        if not gt_label_path.exists():
            print(f"Error: GT label not found at {gt_label_path}")
            return
        if not pred_label_path_1.exists():
            print(f"Error: Predicted label 1 not found at {pred_label_path_1}")
            return
        if not pred_label_path_2.exists():
            print(f"Error: Predicted label 2 not found at {pred_label_path_2}")
            return

        # load coordinates
        scan = np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4)
        coords = scan[:, :3]

        # load raw labels
        gt_labels_raw = np.fromfile(gt_label_path, dtype=np.uint32) & 0xFFFF
        pred_labels_raw_1 = np.fromfile(pred_label_path_1, dtype=np.uint32) & 0xFFFF
        pred_labels_raw_2 = np.fromfile(pred_label_path_2, dtype=np.uint32) & 0xFFFF

        if subsample_rate > 1:
            coords = coords[::subsample_rate]
            gt_labels_raw = gt_labels_raw[::subsample_rate]
            pred_labels_raw_1 = pred_labels_raw_1[::subsample_rate]
            pred_labels_raw_2 = pred_labels_raw_2[::subsample_rate]

        # mapping logic (raw -> 0-18)
        mapper = np.full(260, -1, dtype=np.int32)
        for k, v in LABEL_MAP.items():
            mapper[k] = v

        # map GT and Predictions to 0-18
        gt_mapped = mapper[np.clip(gt_labels_raw, 0, 259)]
        pred_mapped_1 = mapper[np.clip(pred_labels_raw_1, 0, 259)]
        pred_mapped_2 = mapper[np.clip(pred_labels_raw_2, 0, 259)]

        # create the color lookup table for indices 0-18
        lut = np.zeros((20, 3))
        for lbl, col in TRAIN_COLOR_MAP.items():
            idx = 19 if lbl == -1 else lbl
            lut[idx] = col

        gt_colors = lut[np.where(gt_mapped == -1, 19, gt_mapped)]
        pred_colors_1 = lut[np.where(pred_mapped_1 == -1, 19, pred_mapped_1)]
        pred_colors_2 = lut[np.where(pred_mapped_2 == -1, 19, pred_mapped_2)]

        fig = plt.figure(figsize=figsize, facecolor='white', dpi=300)

        # 1. Ground Truth plot (Left)
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.set_facecolor('white')
        ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=gt_colors, s=1.0, edgecolors='none')
        ax1.set_title(f"Ground Truth\n(Seq: {sequence} | Frame: {frame_id})", fontsize=18, fontweight='bold',
                      color='black', pad=15)
        ax1.set_axis_off()

        # 2. First Prediction plot (Middle)
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.set_facecolor('white')
        ax2.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=pred_colors_1, s=1.0, edgecolors='none')
        ax2.set_title(f"Predicted Labels\n[{formatted_subtitle_1}]", fontsize=18, fontweight='bold', color='black',
                      pad=15)
        ax2.set_axis_off()

        # 3. Second Prediction plot (Right)
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.set_facecolor('white')
        ax3.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=pred_colors_2, s=1.0, edgecolors='none')
        ax3.set_title(f"Predicted Labels\n[{formatted_subtitle_2}]", fontsize=18, fontweight='bold', color='black',
                      pad=15)
        ax3.set_axis_off()

        # camera settings & aspect ratio
        for ax in [ax1, ax2, ax3]:
            x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
            y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
            z_min, z_max = coords[:, 2].min(), coords[:, 2].max()

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)

            x_range, y_range, z_range = x_max - x_min, y_max - y_min, z_max - z_min

            # enhanced zoom
            try:
                ax.set_box_aspect((x_range, y_range, z_range), zoom=zoom)
            except TypeError:
                ax.set_box_aspect((x_range, y_range, z_range))
                ax.dist = 4

            # steeper camera tilt
            ax.view_init(elev=elev, azim=azim)
            ax.margins(0)

        # Unique labels across all three plots for the unified legend
        unique_labels_mapped = np.unique(np.concatenate((gt_mapped, pred_mapped_1, pred_mapped_2)))

        legend_elements = []
        for val in unique_labels_mapped:
            if val in PRESENTATION_CLASSES:
                color_idx = 19 if val == -1 else val
                class_name = CLASS_NAMES[val]
                legend_elements.append(mpatches.Patch(color=lut[color_idx], label=class_name))

        # Increased legend font size
        fig.legend(handles=legend_elements, loc='lower center', ncol=len(legend_elements),
                   bbox_to_anchor=(0.5, 0.05), frameon=True, fontsize=12,
                   facecolor='white', edgecolor='lightgray')

        # Reduced wspace to pull the 3D plots closer together
        plt.subplots_adjust(left=0.02, right=0.98, bottom=0.15, top=0.90, wspace=-0.35)

        # save as pdf
        #plt.savefig(f"frame{frame_id}.pdf", format="pdf", bbox_inches="tight", dpi=300)

        plt.show()

        # Explicitly clear the figure from Matplotlib's memory to prevent RAM exhaustion
        plt.close(fig)

    def visualize_sampling_strategies(self, sequence: str, frame_id: str, max_points: int = 10240,
                                      figsize=(22, 7), azim=-60, zoom=1.2, wspace=-0.30):
        """
        Visualizes the three different sampling strategies (block, SFC, kNN) applied to a single frame.
        """
        project_root = self.root_dir.resolve().parent.parent
        data_dir = project_root / "data" / "kitti" / "dataset" / "sequences" / sequence

        velodyne_path = data_dir / "velodyne" / f"{frame_id}.bin"
        gt_label_path = data_dir / "labels" / f"{frame_id}.label"

        if not velodyne_path.exists():
            print(f"Error: Point cloud not found at {velodyne_path}")
            return
        if not gt_label_path.exists():
            print(f"Error: GT label not found at {gt_label_path}")
            return

        # load coordinates and raw labels
        scan = np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4)
        coords = scan[:, :3]
        labels_raw = np.fromfile(gt_label_path, dtype=np.uint32) & 0xFFFF

        # map labels to 0-18 evaluation classes
        mapper = np.full(260, -1, dtype=np.int32)
        for k, v in LABEL_MAP.items():
            mapper[k] = v
        labels_mapped = mapper[np.clip(labels_raw, 0, 259)]

        num_points = coords.shape[0]

        # block
        block_idx = np.array([])
        for _ in range(10):
            center = coords[np.random.randint(num_points)]
            mask = (coords[:, 0] >= center[0] - 4.0) & (coords[:, 0] < center[0] + 4.0) & \
                   (coords[:, 1] >= center[1] - 4.0) & (coords[:, 1] < center[1] + 4.0)
            block_idx = np.nonzero(mask)[0]
            if len(block_idx) >= 1024:
                break
        if len(block_idx) == 0:
            block_idx = np.array([np.random.randint(num_points)])
        if len(block_idx) > max_points:
            np.random.shuffle(block_idx)
            block_idx = block_idx[:max_points]

        # SFC
        min_coord = coords.min(axis=0)
        quantized = ((coords - min_coord) / 0.01).astype(np.uint64)

        x = quantized[:, 0]
        y = quantized[:, 1]
        z = quantized[:, 2]

        max_val = quantized.max()
        bits = int(np.ceil(np.log2(max_val + 1))) if max_val > 0 else 1

        morton_codes = np.zeros(num_points, dtype=np.uint64)
        for i in range(bits):
            morton_codes |= ((x >> i) & 1) << (3 * i)
            morton_codes |= ((y >> i) & 1) << (3 * i + 1)
            morton_codes |= ((z >> i) & 1) << (3 * i + 2)

        sort_idx = np.argsort(morton_codes)

        if num_points > max_points:
            start_idx = np.random.randint(0, num_points - max_points + 1)
            curve_idx = sort_idx[start_idx:start_idx + max_points]
        else:
            curve_idx = sort_idx

        # kNN
        center = coords[np.random.randint(num_points)]
        dists = np.sum((coords - center) ** 2, axis=1)
        knn_idx = np.argpartition(dists, max_points)[:max_points]

        strategies = [
            ("Sliding Block (8x8m)", block_idx),
            ("Space-Filling Curve (Hilbert/Morton)", curve_idx),
            ("k-Nearest Neighbors (k-NN)", knn_idx)
        ]

        lut = np.zeros((20, 3))
        for lbl, col in TRAIN_COLOR_MAP.items():
            idx = 19 if lbl == -1 else lbl
            lut[idx] = col

        fig = plt.figure(figsize=figsize, facecolor='white', dpi=300)
        all_unique_labels = set()

        for i, (title, indices) in enumerate(strategies):
            chunk_coords = coords[indices]
            chunk_labels = labels_mapped[indices]

            all_unique_labels.update(chunk_labels)

            chunk_colors = lut[np.where(chunk_labels == -1, 19, chunk_labels)]

            ax = fig.add_subplot(1, 3, i + 1, projection='3d')
            ax.set_facecolor('white')
            ax.scatter(chunk_coords[:, 0], chunk_coords[:, 1], chunk_coords[:, 2],
                       c=chunk_colors, s=3.0, edgecolors='none')

            ax.set_title(f"{title}\n({len(indices)} points)", fontsize=16, fontweight='bold', color='black')
            ax.set_axis_off()

            x_min, x_max = chunk_coords[:, 0].min(), chunk_coords[:, 0].max()
            y_min, y_max = chunk_coords[:, 1].min(), chunk_coords[:, 1].max()
            z_min, z_max = chunk_coords[:, 2].min(), chunk_coords[:, 2].max()

            center_x = (x_max + x_min) / 2.0
            center_y = (y_max + y_min) / 2.0
            center_z = (z_max + z_min) / 2.0

            max_xy_range = max(x_max - x_min, y_max - y_min) / 2.0
            z_range = max(z_max - z_min, 1.0)

            padding = max_xy_range * 0.20
            max_xy_range += padding

            ax.set_xlim(center_x - max_xy_range, center_x + max_xy_range)
            ax.set_ylim(center_y - max_xy_range, center_y + max_xy_range)
            ax.set_zlim(center_z - z_range / 2.0, center_z + z_range / 2.0)

            try:
                ax.set_box_aspect((max_xy_range * 2, max_xy_range * 2, z_range), zoom=zoom)
            except TypeError:
                ax.set_box_aspect((max_xy_range * 2, max_xy_range * 2, z_range))
                ax.dist = 10 / zoom

            ax.view_init(elev=50, azim=azim)
            ax.margins(0)

        legend_elements = []
        for val in sorted(list(all_unique_labels)):
            if val in PRESENTATION_CLASSES:
                color_idx = 19 if val == -1 else val
                class_name = CLASS_NAMES[val]
                legend_elements.append(mpatches.Patch(color=lut[color_idx], label=class_name))

        fig.legend(handles=legend_elements, loc='lower center', ncol=len(legend_elements),
                   bbox_to_anchor=(0.5, 0.05), frameon=True, fontsize=14,
                   facecolor='white', edgecolor='lightgray')

        plt.subplots_adjust(left=0.0, right=1.0, bottom=0.15, top=0.90, wspace=wspace)
        plt.show()

    def plot_coverage_vs_redundancy(self, selected_evals=None, figsize=(12, 6)):
        """
        Plots the interpolation rate (IP) in % vs oversampling factor (OS-Factor).
        """
        df = self.get_evaluations()
        if df.empty:
            print("No evaluation data available to plot.")
            return

        df_filtered = df[df['Eval'].str.contains('logit_average')].copy()
        df_filtered = df_filtered.dropna(subset=['OS-Factor', 'IP %'])

        if selected_evals:
            df_filtered = df_filtered[df_filtered['Eval'].isin(selected_evals)]

        # sort by interpolation rate
        df_filtered = df_filtered.sort_values(by='IP %', ascending=False)

        if df_filtered.empty:
            print("No matching runs found.")
            return

        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = ax1.twinx()

        x = np.arange(len(df_filtered))
        width = 0.35

        # customized bar labels
        CUSTOM_LABELS = {
            'fps_knn': 'Uniform Anchoring\n(FPS + kNN)\n60 Anchors',
            'voxel_knn': 'Uniform Anchoring\n(Voxel + kNN)\n$8m^3$ Voxel Size',
            'kdtree_knn': 'Density-Adaptive\n(Recursive kd-tree)\n32 Cells',
            'nuc_knn': 'Density-Adaptive\n(NUC)\n$a_0$=2m, $\delta$=1m,\n$B_\\theta$=8, $B_z$=1',
            'block': 'Planar Block\n(Baseline)\n$8m^2$, 0.5m overlap',
            'hilbert': 'Hilbert Curve\n(1D Serialization)'
        }

        def clean_label(eval_name, strategy):
            base_name = CUSTOM_LABELS.get(strategy, strategy.upper())
            params = eval_name.replace(f"{strategy}_logit_average_", "")
            if 'Hilbert' in base_name:
                if '1000' in params:
                    return f"{base_name}\n1k Points Overlap"
                if '5000' in params:
                    return f"{base_name}\n5k Points Overlap"
            else:
                return f"{base_name}"

        labels = [clean_label(row['Eval'], row['Strategy']) for _, row in df_filtered.iterrows()]

        color_os = '#1f77b4'  # blue for oversampling
        color_ip = '#d62728'  # red for missed points

        bars1 = ax1.bar(x - width / 2, df_filtered['OS-Factor'], width, label='Oversampling Factor', color=color_os,
                        alpha=0.85, edgecolor='black')
        bars2 = ax2.bar(x + width / 2, df_filtered['IP %'], width, label='Interpolation Rate (%)', color=color_ip,
                        alpha=0.85, edgecolor='black')

        max_os = df_filtered['OS-Factor'].max()
        max_ip = df_filtered['IP %'].max()
        ax1.set_ylim(0, max_os * 1.10)
        ax2.set_ylim(0, max_ip * 1.10)

        ax1.set_ylabel('Oversampling Factor (x)', color=color_os, fontweight='bold')
        ax2.set_ylabel('Interpolation Rate / Missed Points (%)', color=color_ip, fontweight='bold')

        ax1.tick_params(axis='y', labelcolor=color_os)
        ax2.tick_params(axis='y', labelcolor=color_ip)

        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha='right')

        for bar in bars1:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, yval + (max_os * 0.02), f'{yval:.2f}x', ha='center',
                     va='bottom', color=color_os, fontweight='bold')

        for bar in bars2:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, yval + (max_ip * 0.02), f'{yval:.1f}%', ha='center',
                     va='bottom', color=color_ip, fontweight='bold')

        ax1.grid(axis='y', linestyle='--', alpha=0.4)

        # legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', framealpha=0.9, edgecolor='black')

        plt.tight_layout()

        # save as pdf
        #plt.savefig("tiling_coverage.pdf", format="pdf", bbox_inches="tight")
        plt.show()

    def plot_sampling_latency(self, selected_evals=None, figsize=(14, 7)):
        """
        Plots CPU sampling latency (log scale), GPU model latency (log scale) and oversampling factor (linear).
        """
        df = self.get_evaluations()
        if df.empty:
            print("No evaluation data available to plot.")
            return

        # filter for logit_average and remove NaNs
        df_filtered = df[df['Eval'].str.contains('logit_average')].copy()
        df_filtered = df_filtered.dropna(subset=['OS-Factor', 'Sampling (ms)', 'Model-Lat (ms)'])

        if selected_evals:
            df_filtered = df_filtered[df_filtered['Eval'].isin(selected_evals)]

        # sort by model latency
        df_filtered = df_filtered.sort_values(by='Model-Lat (ms)', ascending=False)

        if df_filtered.empty:
            print("No matching runs found.")
            return

        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = ax1.twinx()

        x = np.arange(len(df_filtered))
        width = 0.25

        CUSTOM_LABELS = {
            'fps_knn': 'Uniform Anchoring\n(FPS + kNN)\n60 Anchors',
            'voxel_knn': 'Uniform Anchoring\n(Voxel + kNN)\n$8m^3$ Voxel Size',
            'kdtree_knn': 'Density-Adaptive\n(Recursive kd-tree)\n32 Cells',
            'nuc_knn': 'Density-Adaptive\n(NUC)\n$a_0$=2m, $\delta$=1m,\n$B_\\theta$=8, $B_z$=1',
            'block': 'Planar Block\n(Baseline)\n$8m^2$, 0.5m overlap',
            'hilbert': 'Hilbert Curve\n(1D Serialization)'
        }

        def clean_label(eval_name, strategy):
            base_name = CUSTOM_LABELS.get(strategy, strategy.upper())
            params = eval_name.replace(f"{strategy}_logit_average_", "")
            if 'Hilbert' in base_name:
                if '1000' in params:
                    return f"{base_name}\n1k Points Overlap"
                if '5000' in params:
                    return f"{base_name}\n5k Points Overlap"
            return f"{base_name}"

        labels = [clean_label(row['Eval'], row['Strategy']) for _, row in df_filtered.iterrows()]

        color_cpu = '#2ca02c'  # green for CPU sampling
        color_gpu = '#ff7f0e'  # orange for GPU model
        color_os = '#1f77b4'  # blue for oversampling

        bars_cpu = ax1.bar(x - width, df_filtered['Sampling (ms)'], width, label='CPU Sampling Latency (ms)',
                           color=color_cpu, alpha=0.85, edgecolor='black')
        bars_gpu = ax1.bar(x, df_filtered['Model-Lat (ms)'], width, label='GPU Model Latency (ms)', color=color_gpu,
                           alpha=0.85, edgecolor='black')

        bars_os = ax2.bar(x + width, df_filtered['OS-Factor'], width, label='Oversampling Factor', color=color_os,
                          alpha=0.85, edgecolor='black')
        ax1.set_yscale('log')
        max_os = df_filtered['OS-Factor'].max()
        ax1.set_ylim(bottom=1)
        ax1.set_ylim(top=df_filtered['Model-Lat (ms)'].max() * 5.0)
        ax2.set_ylim(0, max_os * 1.30)

        ax1.set_ylabel('Latency (ms / frame) [Log Scale]', color='black', fontweight='bold')
        ax2.set_ylabel('Oversampling Factor (x)', color=color_os, fontweight='bold')

        ax1.tick_params(axis='y', labelcolor='black')
        ax2.tick_params(axis='y', labelcolor=color_os)

        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha='right')

        for bar in bars_cpu:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, yval * 1.15, f'{yval:.1f}', ha='center', va='bottom',
                     color=color_cpu, fontweight='bold')

        for bar in bars_gpu:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, yval * 1.15, f'{int(yval):,}', ha='center', va='bottom',
                     color=color_gpu, fontweight='bold')

        x_nudge_offset = 0.03
        for bar in bars_os:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2 + x_nudge_offset, yval + (max_os * 0.02), f'{yval:.2f}x',
                     ha='center', va='bottom', color=color_os, fontweight='bold')

        ax1.grid(axis='y', linestyle='--', alpha=0.4, which='both')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', framealpha=0.9,
                   edgecolor='black')

        plt.tight_layout()

        # save as pdf
        #plt.savefig("sampling_latency.pdf", format="pdf", bbox_inches="tight")
        plt.show()

    def plot_sampling_accuracy(self, selected_evals=None, figsize=(10, 6)):
        """
        Plots a scatter chart of End-to-End latency vs. mIoU for the logit_average runs.
        """
        df = self.get_evaluations()
        if df.empty:
            print("No evaluation data available to plot.")
            return

        # filter for logit_average
        df_filtered = df[df['Eval'].str.contains('logit_average')].copy()
        df_filtered = df_filtered.dropna(subset=['E2E-Lat (ms)', 'mIoU (%)'])

        if selected_evals:
            df_filtered = df_filtered[df_filtered['Eval'].isin(selected_evals)]

        if df_filtered.empty:
            print("No matching runs found.")
            return

        fig, ax = plt.subplots(figsize=figsize)

        CUSTOM_LABELS = {
            'fps_knn': 'Uniform (FPS based)',
            'voxel_knn': 'Uniform (Voxel based)',
            'kdtree_knn': 'KD-Tree',
            'nuc_knn': 'NUC',
            'block': 'Planar Block',
            'hilbert': 'Hilbert'
        }

        def clean_label(eval_name, strategy):
            base_name = CUSTOM_LABELS.get(strategy, strategy.upper())
            params = eval_name.replace(f"{strategy}_logit_average_", "")

            if 'Hilbert' in base_name:
                if '1000' in params: return f"{base_name} (1k Points Overlap)"
                if '5000' in params: return f"{base_name} (5k Points Overlap)"
            return base_name

        strategies = df_filtered["Strategy"].unique()

        for strategy in strategies:
            subset = df_filtered[df_filtered["Strategy"] == strategy]
            ax.scatter(
                subset["E2E-Lat (ms)"],
                subset["mIoU (%)"],
                color='black',
                s=80,
                alpha=0.85,
                edgecolors='black',
                linewidth=1.5
            )

            for _, row in subset.iterrows():
                label_text = clean_label(row['Eval'], row['Strategy'])

                offset = (8, 8)

                if 'Hilbert' in label_text and '1k' in label_text:
                    offset = (7, -18)
                elif 'Voxel' in label_text:
                    offset = (-52, 12)

                ax.annotate(
                    label_text,
                    (row["E2E-Lat (ms)"], row["mIoU (%)"]),
                    xytext=offset,
                    textcoords='offset points',
                    fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none")
                )

        ax.set_xlabel("End-to-End Latency (ms / frame)", fontweight='bold')
        ax.set_ylabel("Mean Intersection over Union (mIoU %)", fontweight='bold')

        ax.tick_params(axis='both')
        ax.grid(True, linestyle='--', alpha=0.5)

        ax.set_xlim(df_filtered["E2E-Lat (ms)"].min() * 0.5, df_filtered["E2E-Lat (ms)"].max() * 1.15)
        ax.set_ylim(df_filtered["mIoU (%)"].min() * 0.95, df_filtered["mIoU (%)"].max() * 1.05)
        plt.tight_layout()

        # save as pdf
        #plt.savefig("sampling_accuracy.pdf", format="pdf", bbox_inches="tight")
        plt.show()

    def plot_fusion(self, selected_evals=None, figsize=(14, 7)):
        """
        Plots a grouped dual-axis bar chart comparing ALP vs. MC Uncertainty fusion.
        """
        df = self.get_evaluations()
        if df.empty:
            print("No evaluation data available to plot.")
            return

        if selected_evals:
            df_filtered = df[df['Eval'].isin(selected_evals)].copy()
        else:
            print("Please provide a list of selected_evals for this plot.")
            return

        if df_filtered.empty:
            print("No matching runs found.")
            return

        def get_base_strategy(eval_name):
            if 'kdtree' in eval_name: return 'KD-Tree Tiling'
            if 'nuc' in eval_name: return 'NUC Tiling'
            if 'hilbert' in eval_name and '1000' in eval_name: return 'Hilbert (1k) Tiling'
            if 'hilbert' in eval_name and '5000' in eval_name: return 'Hilbert (5k) Tiling'
            return eval_name

        def get_fusion_type(eval_name):
            if 'logit_average' in eval_name: return 'Average Logit Fusion'
            if 'mc_uncertainty' in eval_name: return 'MC Dropout based Fusion'
            return 'Unknown'

        df_filtered['Base_Strategy'] = df_filtered['Eval'].apply(get_base_strategy)
        df_filtered['Fusion_Type'] = df_filtered['Eval'].apply(get_fusion_type)

        cat_order = ['NUC Tiling', 'KD-Tree Tiling', 'Hilbert (1k) Tiling', 'Hilbert (5k) Tiling']
        df_filtered['Base_Strategy'] = pd.Categorical(df_filtered['Base_Strategy'], categories=cat_order, ordered=True)
        df_filtered = df_filtered.sort_values(['Base_Strategy', 'Fusion_Type'])

        pivot_miou = df_filtered.pivot_table(index='Base_Strategy', columns='Fusion_Type', values='mIoU (%)',
                                             aggfunc='mean')
        pivot_lat = df_filtered.pivot_table(index='Base_Strategy', columns='Fusion_Type', values='E2E-Lat (ms)',
                                            aggfunc='mean')

        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = ax1.twinx()

        x = np.arange(len(cat_order))
        width = 0.18

        color_miou = '#2ca02c'  # green for accuracy
        color_lat = '#d62728'  # red for latency

        bars_miou_logit = ax1.bar(x - width * 1.5, pivot_miou['Average Logit Fusion'], width,
                                  label='mIoU (Average Logit Fusion)',
                                  color=color_miou, alpha=0.9, edgecolor='black')
        bars_miou_mc = ax1.bar(x - width * 0.5, pivot_miou['MC Dropout based Fusion'], width,
                               label='mIoU (MC Dropout based Fusion)',
                               color=color_miou, alpha=0.4, edgecolor='black', hatch='//')

        bars_lat_logit = ax2.bar(x + width * 0.5, pivot_lat['Average Logit Fusion'], width,
                                 label='Latency (Average Logit Fusion)',
                                 color=color_lat, alpha=0.9, edgecolor='black')
        bars_lat_mc = ax2.bar(x + width * 1.5, pivot_lat['MC Dropout based Fusion'], width,
                              label='Latency (MC Dropout based Fusion)',
                              color=color_lat, alpha=0.4, edgecolor='black', hatch='//')

        max_miou = pivot_miou.max().max()
        ax1.set_ylim(0, max_miou * 1.35)

        max_lat = pivot_lat.max().max()
        ax2.set_yscale('log')
        ax2.set_ylim(bottom=1)
        ax2.set_ylim(top=max_lat * 10.0)

        ax1.set_ylabel('Mean Intersection over Union (mIoU %)', color=color_miou, fontweight='bold')
        ax2.set_ylabel('End-to-End Latency (ms) [Log Scale]', color=color_lat, fontweight='bold')

        ax1.tick_params(axis='y', labelcolor=color_miou)
        ax2.tick_params(axis='y', labelcolor=color_lat)

        ax1.set_xticks(x)
        ax1.set_xticklabels(cat_order, fontweight='bold')

        text_angle = 35
        x_nudge = -0.06

        for bar in bars_miou_logit + bars_miou_mc:
            yval = bar.get_height()
            if not np.isnan(yval):
                ax1.text(bar.get_x() + bar.get_width() / 2 + x_nudge, yval + (max_miou * 0.02), f'{yval:.2f}%',
                         ha='left', va='bottom', color=color_miou, fontweight='bold', rotation=text_angle)

        for bar in bars_lat_logit + bars_lat_mc:
            yval = bar.get_height()
            if not np.isnan(yval):
                ax2.text(bar.get_x() + bar.get_width() / 2 + x_nudge, yval * 1.20, f'{int(yval):,}',
                         ha='left', va='bottom', color=color_lat, fontweight='bold', rotation=text_angle)

        ax1.grid(axis='y', linestyle='--', alpha=0.4)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()

        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', framealpha=0.9,
                   edgecolor='black', ncol=2)

        plt.tight_layout()

        # save as pdf
        #plt.savefig("fusion.pdf", format="pdf", bbox_inches="tight")
        plt.show()

    def plot_pipeline(self, selected_evals=None, figsize=(14, 7)):
        """
        Plots a grouped bar chart on a logarithmic scale showing the Anatomy of a Frame.
        """
        df = self.get_evaluations()
        if df.empty:
            print("No evaluation data available to plot.")
            return

        if selected_evals:
            df_filtered = df[df['Eval'].isin(selected_evals)].copy()
        else:
            print("Please provide a list of selected_evals.")
            return

        if df_filtered.empty:
            print("No matching runs found.")
            return

        CUSTOM_LABELS = {
            'voxel_knn_mc_uncertainty_8': 'Voxel kNN\n(MC Dropout Fusion)',
            'block_logit_average_8_05': 'Planar Block\n(AVG Logit Fusion)',
            'nuc_knn_logit_average_2181': 'NUC Tiling\n(AVG Logit Fusion)',
            'hilbert_mc_uncertainty_o1000': 'Hilbert 1k\n(MC Dropout Fusion)',
            'hilbert_logit_average_o1000': 'Hilbert 1k\n(AVG Logit Fusion)'
        }

        order = list(CUSTOM_LABELS.keys())
        df_filtered['Eval'] = pd.Categorical(df_filtered['Eval'], categories=order, ordered=True)
        df_filtered = df_filtered.sort_values('Eval')

        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(df_filtered))
        width = 0.25

        color_sample = '#2ca02c'  # green
        color_model = '#ff7f0e'  # orange
        color_fusion = '#9467bd'  # purple

        bars_sample = ax.bar(x - width, df_filtered['Sampling (ms)'], width, label='Tiling Latency (CPU)',
                             color=color_sample, alpha=0.9, edgecolor='black')
        bars_model = ax.bar(x, df_filtered['Model-Lat (ms)'], width, label='Model Latency (GPU)', color=color_model,
                            alpha=0.9, edgecolor='black')
        bars_fusion = ax.bar(x + width, df_filtered['Fusion (ms)'], width, label='Fusion Latency (CPU/GPU)',
                             color=color_fusion, alpha=0.9, edgecolor='black')

        ax.set_yscale('log')
        ax.set_ylim(bottom=1)

        max_val = df_filtered[['Sampling (ms)', 'Model-Lat (ms)', 'Fusion (ms)']].max().max()
        ax.set_ylim(top=max_val * 20.0)

        ax.set_ylabel('Latency (ms) [Log Scale]', fontsize=16, color='black', fontweight='bold')
        ax.tick_params(axis='y', labelsize=14)

        labels = [CUSTOM_LABELS[val] for val in df_filtered['Eval']]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=16, fontweight='bold')

        text_angle = 90

        for bars, color in zip([bars_sample, bars_model, bars_fusion], [color_sample, color_model, color_fusion]):
            for bar in bars:
                yval = bar.get_height()
                if not np.isnan(yval):
                    if yval < 100:
                        text_str = f'{yval:.1f} ms'
                    else:
                        text_str = f'{int(yval):,} ms'

                    ax.text(bar.get_x() + bar.get_width() / 2, yval * 1.25, text_str,
                            ha='center', va='bottom', color=color, fontsize=12, fontweight='bold',
                            rotation=text_angle)

        ax.grid(axis='y', linestyle='--', alpha=0.4, which='both')

        ax.legend(loc='upper right', fontsize=13, framealpha=0.9, edgecolor='black')

        plt.tight_layout()

        # save as pdf
        #plt.savefig("pipeline_latency.pdf", format="pdf", bbox_inches="tight")
        plt.show()

    def plot_training_scalability(self, experiment_folders, figsize=(10, 6)):
        """
        Plots a scaling curve comparing actual training throughput vs. ideal linear scaling.
        """
        import json
        from pathlib import Path
        import matplotlib.pyplot as plt

        current_file_path = Path(__file__).resolve()
        base_dir = current_file_path.parent.parent.parent / "notebooks" / "experiments"

        throughput_data = {}

        # parse metrics
        for folder_name in experiment_folders:
            try:
                node_part = [part for part in folder_name.split('_') if 'nodes' in part][0]
                node_count = int(node_part.replace('nodes', ''))
            except (IndexError, ValueError):
                print(f"Could not parse node count from folder name: {folder_name}")
                continue

            metrics_file = base_dir / folder_name / "metrics.jsonl"
            if not metrics_file.exists():
                print(f"Metrics file not found: {metrics_file}")
                continue

            fps_values = []
            with open(metrics_file, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line.strip())
                    if "throughput_fps" in data:
                        fps_values.append(data["throughput_fps"])

            if not fps_values:
                print(f"No throughput_fps data found in {metrics_file}")
                continue

            avg_fps = sum(fps_values) / len(fps_values)
            throughput_data[node_count] = avg_fps

        if not throughput_data:
            print("No valid throughput data parsed. Cannot plot.")
            return

        nodes = sorted(list(throughput_data.keys()))
        actual_fps = [throughput_data[n] for n in nodes]

        base_fps = actual_fps[0]
        base_node_count = nodes[0]
        ideal_fps = [(base_fps / base_node_count) * n for n in nodes]
        efficiency = [(actual / ideal) * 100 for actual, ideal in zip(actual_fps, ideal_fps)]

        fig, ax = plt.subplots(figsize=figsize)

        color_actual = '#1f77b4'  # blue
        color_ideal = 'grey'  # grey for the theoretical baseline

        ax.plot(nodes, ideal_fps, marker='s', linestyle='--', color=color_ideal,
                linewidth=2, markersize=8, label='Ideal Linear Scaling')

        ax.plot(nodes, actual_fps, marker='o', linestyle='-', color=color_actual,
                linewidth=3, markersize=10, label='Actual System Throughput')

        ax.fill_between(nodes, actual_fps, ideal_fps, color='grey', alpha=0.1)

        ax.set_ylim(0, max(ideal_fps) * 1.25)
        ax.set_xlim(min(nodes) - 0.2, max(nodes) + 0.2)
        ax.set_ylabel('Throughput in Frames per Second (FPS)', fontweight='bold')
        ax.set_xlabel('Number of Active Worker Nodes', fontweight='bold')

        ax.set_xticks(nodes)
        ax.set_xticklabels([f'{n} Node{"s" if n > 1 else ""}' for n in nodes])

        ax.tick_params(axis='y')
        ax.grid(True, linestyle='--', alpha=0.6)

        for i, (n, a_fps, i_fps, eff) in enumerate(zip(nodes, actual_fps, ideal_fps, efficiency)):
            text_color = 'black' if eff > 80 else '#d62728'
            ax.text(n, a_fps - (max(ideal_fps) * 0.08), f'{a_fps:.2f} FPS\n({eff:.1f}%)',
                    ha='center', va='top', color=text_color, fontweight='bold')

            if i > 0:
                ax.text(n, i_fps + (max(ideal_fps) * 0.03), f'{i_fps:.2f} FPS',
                        ha='center', va='bottom', color='dimgrey', fontweight='bold')

        ax.legend(loc='upper left', framealpha=0.9, edgecolor='black')
        plt.tight_layout()

        # save as pdf
        #plt.savefig("training_scalability.pdf", format="pdf", bbox_inches="tight")
        plt.show()

    def plot_inference_scalability(self, experiment_id, eval_prefix, node_counts=[1, 2, 3], figsize=(10, 6)):
        """
        Plots a scaling curve comparing actual inference throughput vs. ideal linear scaling.
        """
        import json
        from pathlib import Path
        import matplotlib.pyplot as plt

        current_file_path = Path(__file__).resolve()
        inference_dir = current_file_path.parent.parent.parent / "notebooks" / "experiments" / experiment_id / "inference"

        throughput_data = {}

        # parse metrics
        for nodes in node_counts:
            folder_name = f"{eval_prefix}_{nodes}nodes"
            metrics_file = inference_dir / folder_name / "evaluation_metrics.json"

            if not metrics_file.exists():
                print(f"Metrics file not found: {metrics_file}")
                continue

            with open(metrics_file, 'r') as f:
                data = json.load(f)

            fps = data.get("results", {}).get("cluster_throughput_fps")
            if fps is not None:
                throughput_data[nodes] = fps
            else:
                print(f"No cluster_throughput_fps found in {metrics_file}")

        if not throughput_data:
            print("No valid throughput data parsed. Cannot plot.")
            return

        nodes_sorted = sorted(list(throughput_data.keys()))
        actual_fps = [throughput_data[n] for n in nodes_sorted]

        base_fps = actual_fps[0]
        base_node_count = nodes_sorted[0]
        ideal_fps = [(base_fps / base_node_count) * n for n in nodes_sorted]

        efficiency = [(actual / ideal) * 100 for actual, ideal in zip(actual_fps, ideal_fps)]

        fig, ax = plt.subplots(figsize=figsize)

        color_actual = '#2ca02c'  # green
        color_ideal = 'grey'  # grey for the theoretical baseline

        ax.plot(nodes_sorted, ideal_fps, marker='s', linestyle='--', color=color_ideal,
                linewidth=2, markersize=8, label='Ideal Linear Scaling')

        ax.plot(nodes_sorted, actual_fps, marker='o', linestyle='-', color=color_actual,
                linewidth=3, markersize=10, label='Actual System Throughput')

        ax.fill_between(nodes_sorted, actual_fps, ideal_fps, color='grey', alpha=0.1)

        ax.set_ylim(0, max(ideal_fps) * 1.25)
        ax.set_xlim(min(nodes_sorted) - 0.2, max(nodes_sorted) + 0.2)
        ax.set_ylabel('Throughput in Frames per Second (FPS)', fontweight='bold')
        ax.set_xlabel('Number of Active Worker Nodes', fontweight='bold')

        ax.set_xticks(nodes_sorted)
        ax.set_xticklabels([f'{n} Node{"s" if n > 1 else ""}' for n in nodes_sorted])

        ax.tick_params(axis='y')
        ax.grid(True, linestyle='--', alpha=0.6)

        for i, (n, a_fps, i_fps, eff) in enumerate(zip(nodes_sorted, actual_fps, ideal_fps, efficiency)):
            ax.text(n, a_fps - (max(ideal_fps) * 0.08), f'{a_fps:.2f} FPS\n({eff:.1f}%)',
                    ha='center', va='top', color='black', fontweight='bold')

            if i > 0:
                ax.text(n, i_fps + (max(ideal_fps) * 0.03), f'{i_fps:.2f} FPS',
                        ha='center', va='bottom', color='dimgrey', fontweight='bold')

        ax.legend(loc='upper left', framealpha=0.9, edgecolor='black')
        plt.tight_layout()

        # save as pdf
        #plt.savefig("inference_scalability.pdf", format="pdf", bbox_inches="tight")
        plt.show()