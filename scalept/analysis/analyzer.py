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

            # Take the most recent evaluation
            latest_eval = data["evaluations"][-1]
            params = latest_eval.get("parameters", {})
            results = latest_eval.get("results", {})

            rows.append({
                "Experiment ID": exp_name,
                "Strategy": params.get("sampling_strategy", "N/A"),
                "mIoU (%)": results.get("mean_iou", np.nan),
                "mAcc (%)": results.get("mean_accuracy", np.nan),
                "OA (%)": results.get("overall_accuracy", np.nan),
                "Latency (ms)": results.get("average_latency_ms", np.nan)
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(by="mIoU (%)", ascending=False).reset_index(drop=True)
        return df

    def get_evaluation(self, id: str) -> Styler:
        rows = []
        for exp_name, data in self.experiments.items():
            if not data["evaluations"] and exp_name != id:
                continue

            # Take most recent evaluation
            latest_eval = data["evaluations"][-1]
            results = latest_eval.get("results", {})
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
        df_cap = df.style.set_caption(f"<h4><b>{id.split('_', 2)[2]}</b></h4>")

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

    def plot_training_loss(self, experiment_ids: list = None, figsize=(10, 5)):
        """
        Plots the training loss curves over epochs for convergence analysis.
        """
        if not experiment_ids:
            experiment_ids = list(self.experiments.keys())

        plt.figure(figsize=figsize)

        for exp_id in experiment_ids:
            if exp_id not in self.experiments or not self.experiments[exp_id]["training"]:
                continue

            train_data = self.experiments[exp_id]["training"]
            epochs = [entry["epoch"] for entry in train_data]
            losses = [entry["train_loss"] for entry in train_data]

            strategy = exp_id
            if self.experiments[exp_id]["evaluations"]:
                strategy = self.experiments[exp_id]["evaluations"][-1]["parameters"].get("sampling_strategy", exp_id)

            plt.plot(epochs, losses, marker='o', linewidth=2, label=f"{strategy.upper()} ({exp_id[-6:]})")

        plt.title("Training Loss Convergence", fontsize=14, fontweight='bold')
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Cross Entropy Loss", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()