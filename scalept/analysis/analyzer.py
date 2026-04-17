import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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

                #if eval_name not in ['overlap2000_knn_seq08', 'overlap2000_hilbert_seq08']:
                if "40ep" not in model_name:
                    rows.append({
                        "Model": model_name,
                        "Eval": eval_name,
                        "Strategy": params.get("sampling_strategy", "N/A"),
                        "mIoU (%)": results.get("mean_iou", np.nan),
                        "mAcc (%)": results.get("mean_accuracy", np.nan),
                        "OA (%)": results.get("overall_accuracy", np.nan),
                        "Sampl-Lat (ms)": results.get("sampling_latency_ms", np.nan),
                        "Model-Lat (ms)": results.get("model_latency_ms", np.nan),
                        "Fusion-Lat (ms)": results.get("fusion_latency_ms", np.nan),
                        "E2E-Lat (ms)": results.get("total_e2e_latency_ms", np.nan),
                        "Cluster Time (min)": results.get("total_cluster_time_min", np.nan),
                        "OS-Factor": results.get("average_oversampling_factor", np.nan),
                        "IP %": results.get("average_interpolated_percentage", np.nan),
                        "Cleanup (avg)": results.get("average_cleanup_iterations", np.nan),
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

    def plot_speed_vs_accuracy(self, figsize=(8, 5)):
        """
        Plots a scatter chart of Inference Latency vs mIoU.
        """
        df = self.get_evaluation_summary_table()
        if df.empty:
            print("No evaluation data available to plot.")
            return

        plt.figure(figsize=figsize)

        strategies = df["Strategy"].unique()
        colors = plt.cm.get_cmap("Set1", len(strategies))

        for i, strategy in enumerate(strategies):
            subset = df[df["Strategy"] == strategy]
            plt.scatter(
                subset["Latency (ms)"],
                subset["mIoU (%)"],
                label=strategy.upper(),
                color=colors(i),
                s=150, alpha=0.8, edgecolors='k'
            )

            for _, row in subset.iterrows():
                plt.annotate(
                    row["Experiment ID"][-6:],
                    (row["Latency (ms)"], row["mIoU (%)"]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9
                )

        plt.title("Speed vs. Accuracy Trade-off", fontsize=14, fontweight='bold')
        plt.xlabel("Inference Latency (ms / frame) $\\rightarrow$ Lower is better", fontsize=12)
        plt.ylabel("Mean IoU (%) $\\rightarrow$ Higher is better", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(title="Sampling Strategy")
        plt.tight_layout()
        plt.show()

    def plot_classwise_iou(self, experiment_ids: list = None, figsize=(14, 6)):
        """
        Plots a grouped bar chart for class-wise IoU. (Hilbert vs KNN)
        """
        if not experiment_ids:
            # Default to the first two experiments if none provided
            experiment_ids = list(self.experiments.keys())[:2]

        n_exps = len(experiment_ids)
        bar_width = 0.8 / n_exps
        indices = np.arange(len(CLASS_NAMES))

        plt.figure(figsize=figsize)

        for i, exp_id in enumerate(experiment_ids):
            if exp_id not in self.experiments or not self.experiments[exp_id]["evaluations"]:
                continue

            latest_eval = self.experiments[exp_id]["evaluations"][-1]
            per_class_iou_dict = latest_eval["results"].get("per_class_iou", {})

            # Ensure order matches CLASS_NAMES
            y_values = [per_class_iou_dict.get(str(idx), 0) for idx in range(len(CLASS_NAMES))]

            strategy = latest_eval["parameters"].get("sampling_strategy", exp_id)
            label = f"{strategy.upper()} ({exp_id[-6:]})"

            plt.bar(indices + i * bar_width, y_values, bar_width, label=label, alpha=0.85)

        plt.title("Per-Class Intersection over Union (IoU)", fontsize=14, fontweight='bold')
        plt.xlabel("Semantic Classes", fontsize=12)
        plt.ylabel("IoU (%)", fontsize=12)

        # Center the x-ticks
        plt.xticks(indices + bar_width * (n_exps - 1) / 2, CLASS_NAMES, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def plot_training_loss(self, experiment_ids: list = None, figsize=(12, 6)):
        """
        Plots the training loss and learning rate curves over epochs.
        Loss is plotted on the left y-axis (solid lines) and learning Rate on the right y-axis (dotted lines).
        """
        if not experiment_ids:
            experiment_ids = list(self.experiments.keys())

        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = ax1.twinx()

        cmap = plt.get_cmap("tab10")

        lines_loss = []
        lines_lr = []

        for i, exp_id in enumerate(experiment_ids):
            if exp_id not in self.experiments or not self.experiments[exp_id]["training"]:
                continue

            train_data = self.experiments[exp_id]["training"]
            epochs = [entry["epoch"] for entry in train_data]
            losses = [entry["train_loss"] for entry in train_data]
            lrs = [entry["learning_rate"] for entry in train_data]

            parts = exp_id.rsplit('_', 2)
            if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
                display_name = parts[0]
            else:
                display_name = exp_id

            color = cmap(i % 10)

            l1, = ax1.plot(epochs, losses, color=color, marker='o', markersize=4,
                           linestyle='-', linewidth=2, label=f"{display_name} (Loss)")
            lines_loss.append(l1)

            l2, = ax2.plot(epochs, lrs, color=color, marker='',
                           linestyle=':', linewidth=2.5, alpha=0.8, label=f"{display_name} (LR)")
            lines_lr.append(l2)

        ax1.set_title("Training Loss and Learning Rate Convergence", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Epoch", fontsize=12)

        ax1.set_ylabel("Cross Entropy Loss", fontsize=12)
        ax2.set_ylabel("Learning Rate", fontsize=12)

        ax1.grid(True, linestyle='--', alpha=0.6)

        lines = lines_loss + lines_lr
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='center left', bbox_to_anchor=(1.12, 0.5), fontsize=10)

        plt.tight_layout()
        plt.show()

    def visualize_frame(self, experiment_id: str, inference_id: str, sequence: str, frame_id: str,
                        figsize=(20, 10), subsample_rate=1):
        """
        Visualizes a specific point cloud frame, comparing Ground Truth vs. Predicted labels.

        Args:
            experiment_id: Name of the experiment directory (e.g., 'hilbert_100ep')
            inference_id: Name of the evaluation run (e.g., 'eval_hilbert_seq08_2000_1')
            sequence: Sequence ID as string (e.g., '08')
            frame_id: Frame number as string (e.g., '000000')
            figsize: Tuple for the matplotlib figure size
            subsample_rate: Integer to plot every Nth point. Set higher (e.g., 5 or 10)
                            if Jupyter notebook hangs during 3D plotting.
        """
        project_root = self.root_dir.resolve().parent.parent
        data_dir = project_root / "data" / "kitti" / "dataset" / "sequences" / sequence

        velodyne_path = data_dir / "velodyne" / f"{frame_id}.bin"
        gt_label_path = data_dir / "labels" / f"{frame_id}.label"

        pred_label_path = self.root_dir / experiment_id / "inference" / inference_id / "predictions" / sequence / f"{frame_id}.label"

        # validations
        if not velodyne_path.exists():
            print(f"Error: Point cloud not found at {velodyne_path}")
            return
        if not gt_label_path.exists():
            print(f"Error: GT label not found at {gt_label_path}")
            return
        if not pred_label_path.exists():
            print(f"Error: Predicted label not found at {pred_label_path}")
            return

        # load coordinates
        scan = np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4)
        coords = scan[:, :3]

        # load labels
        gt_labels = np.fromfile(gt_label_path, dtype=np.uint32) & 0xFFFF
        pred_labels = np.fromfile(pred_label_path, dtype=np.uint32) & 0xFFFF

        if subsample_rate > 1:
            coords = coords[::subsample_rate]
            gt_labels = gt_labels[::subsample_rate]
            pred_labels = pred_labels[::subsample_rate]

        # color mapping
        max_kitti_id = max(KITTI_COLOR_MAP.keys())
        lut = np.zeros((max_kitti_id + 1, 3))
        for lbl, col in KITTI_COLOR_MAP.items():
            lut[lbl] = col

        gt_labels_clipped = np.clip(gt_labels, 0, max_kitti_id)
        pred_labels_clipped = np.clip(pred_labels, 0, max_kitti_id)

        gt_colors = lut[gt_labels_clipped]
        pred_colors = lut[pred_labels_clipped]

        # 300 dpi
        fig = plt.figure(figsize=figsize, facecolor='white', dpi=300)

        # ground Truth
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.set_facecolor('white')
        # at 300 DPI, s=0.5 might look a bit small, bump s up to 1.0 or 2.0 if needed
        ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=gt_colors, s=1.0, edgecolors='none')
        ax1.set_title(f"Ground Truth (Seq: {sequence} | Frame: {frame_id})", fontsize=14, fontweight='bold',
                      color='black')
        ax1.set_axis_off()

        # predictions
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.set_facecolor('white')
        ax2.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=pred_colors, s=1.0, edgecolors='none')
        ax2.set_title(f"Predictions ({experiment_id})", fontsize=14, fontweight='bold', color='black')
        ax2.set_axis_off()

        for ax in [ax1, ax2]:
            x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
            y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
            z_min, z_max = coords[:, 2].min(), coords[:, 2].max()

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)

            x_range = x_max - x_min
            y_range = y_max - y_min
            z_range = z_max - z_min

            # set realistic aspect ratio and force the camera to zoom in
            try:
                ax.set_box_aspect((x_range, y_range, z_range), zoom=1.8)
            except TypeError:
                ax.set_box_aspect((x_range, y_range, z_range))
                ax.dist = 6  # Default is 10, lower numbers zoom the camera in

            ax.view_init(elev=50, azim=-45)  # steeper angle to see the roads better
            ax.margins(0)

        plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.92, wspace=0.0)

        plt.show()