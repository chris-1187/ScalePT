import os
from pathlib import Path
from fabric import Connection, ThreadingGroup
from omegaconf import OmegaConf
from rich.console import Console

console = Console(force_jupyter=False)


class ClusterOrchestrator:

    def __init__(self):
        current_file_path = Path(__file__).resolve()
        self.project_root = current_file_path.parent.parent.parent
        config_dir = self.project_root / "config"
        config_path = config_dir / "config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Critical: Config not found at {config_path}")

        self.cfg = OmegaConf.load(config_path)

        self.base_dir = f"{self.cfg.primary_node.nfs_root}/{self.cfg.args.dataset}"
        self.remote_dataset_dir = f"{self.base_dir}/dataset"
        self.remote_experiments_dir = f"{self.base_dir}/experiments"

        self.cfg.paths.local_worker_repo = self._resolve_local_path(self.cfg.paths.local_worker_repo)
        self.cfg.paths.local_dataset = self._resolve_local_path(self.cfg.paths.local_dataset)
        if 'local_experiments' in self.cfg.paths:
            self.cfg.paths.local_experiments = self._resolve_local_path(self.cfg.paths.local_experiments)
        self.user = self.cfg.project.user

        # Define connection group for all workers
        self.worker_ips = [node.ip for node in self.cfg.worker_nodes]
        self.group = ThreadingGroup(
            *self.worker_ips,
            user=self.user,
            connect_kwargs={"key_filename": self.cfg.project.key_filename}
        )

        # Define primary connection for NFS operations
        self.primary = Connection(
            host=self.cfg.primary_node.ip,
            user=self.user,
            connect_kwargs={"key_filename": self.cfg.project.key_filename}
        )

        self.check_connectivity()

    def run_distributed_training(self, sampling_strategy: str = 'hilbert', run_name: str = ""):
        """
        Launches torchrun on all worker nodes using CUDA 12.1.
        """
        console.rule("[bold]Launching Distributed Training (DDP)")

        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        suffix = f"_{run_name}" if run_name else ""
        exp_folder_name = f"{timestamp}_{sampling_strategy}{suffix}"
        master_addr = self.cfg.primary_node.ip
        master_port = "29500"
        nnodes = self.cfg.args.nnodes_used
        nproc_per_node = self.cfg.args.nproc_per_node

        experiment_dir = f"{self.remote_experiments_dir}/{exp_folder_name}"
        self.primary.run(f"mkdir -p {experiment_dir}/logs")
        self.primary.run(f"mkdir -p {experiment_dir}/weights")

        console.print(f"[magenta]Sampling Strategy: {sampling_strategy}[/magenta]")

        # Command Construction
        cmd_template = (
            "nohup bash -c '"
            "source ~/miniconda3/etc/profile.d/conda.sh && "
            "conda activate spt-worker && "
            "export PATH=/usr/local/cuda-12.1/bin:$PATH && "
            "export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH && "
            "cd ~/spt-worker && "
            # Launch
            "torchrun "
            f"--nproc_per_node={nproc_per_node} "
            f"--nnodes={nnodes} "
            "--node_rank={rank} "
            f"--master_addr={master_addr} "
            f"--master_port={master_port} "
            "-m spt_worker.train "
            f"--data_path {self.remote_dataset_dir} "
            f"--labels_path {self.remote_dataset_dir} "
            f"--output_dir {experiment_dir} "
            f"--sampling_strategy {sampling_strategy} "
            "--sequences 00 01 02 03 04 05 06 07 09 10 "
            "--accumulation_steps 4 "
            "--num_workers 2 "
            "--epochs 40 "
            # Redirect output to a specific log file for this rank
            ">> {experiment_dir}/logs/train_{timestamp}_rank{rank}.log 2>&1"
            "' > /dev/null 2>&1 &"  # Detach
        )

        from fabric import Connection

        console.print(f"[yellow]Master Node: {master_addr}[/yellow]")

        for i, node in enumerate(self.cfg.worker_nodes):
            rank = i
            node_ip = node.ip

            cmd = cmd_template.format(
                rank=rank,
                experiment_dir=experiment_dir,
                timestamp=timestamp
            )

            console.print(f"[dim]Launching Rank {rank} on {node_ip}...[/dim]")

            conn = Connection(
                host=node_ip,
                user=self.user,
                connect_kwargs={"key_filename": self.cfg.project.key_filename}
            )

            conn.run(cmd, hide=True)

        console.print(f"[bold green]✓ All workers launched! Logs available at:[/bold green]")
        console.print(f"[yellow]Weights, metrics and logs available at: {experiment_dir}[/yellow]")

    def run_evaluation(self, experiment_path_relative=None, chunk_size=40000, overlap=2000, sampling_strategy='hilbert',
                       sequences='08'):
        """
        Runs evaluation on the Primary Node (Host 1).
        If no path provided, tries to find the most recent experiment.
        """
        console.rule("[bold]Launching Inference Evaluation")

        if experiment_path_relative:
            remote_exp_dir = f"{self.remote_experiments_dir}/{experiment_path_relative}"
        else:
            cmd_find = f"ls -td {self.remote_experiments_dir}/*/ | head -1"
            res = self.primary.run(cmd_find, hide=True)
            remote_exp_dir = res.stdout.strip()
            if not remote_exp_dir:
                console.print("[red]No experiments found![/red]")
                return

        console.print(f"[yellow]Evaluating Experiment: {remote_exp_dir}[/yellow]")

        # Check for weights
        weights_path = f"{remote_exp_dir.rstrip('/')}/weights/model_weights.pt"
        if self.primary.run(f"[ -f {weights_path} ]", warn=True, hide=True).failed:
            console.print(f"[red]No model_weights.pt found at {weights_path}[/red]")
            return

        inference_id = f"eval_overlap{overlap}_{sampling_strategy}_seq{sequences}"

        # Construct Inference Command
        # Sequence 08 (Validation) by default
        cmd = (
            "source ~/miniconda3/etc/profile.d/conda.sh && "
            "conda activate spt-worker && "
            "export PATH=/usr/local/cuda-12.1/bin:$PATH && "
            "cd ~/spt-worker && "
            "python -m spt_worker.inference "
            f"--data_path {self.remote_dataset_dir} "
            f"--labels_path {self.remote_dataset_dir} "
            f"--checkpoint_path {weights_path} "
            f"--output_dir {remote_exp_dir} "
            f"--inference_id {inference_id} "
            f"--sequences {sequences} "
            f"--num_workers 4 "
            f"--chunk_size {chunk_size} "
            f"--overlap_size {overlap} "
            f"--sampling_strategy {sampling_strategy}"
        )

        # Run on Primary Node
        console.print(f"[dim]Running inference on {self.cfg.primary_node.ip}...[/dim]") # TODO: distributed inference
        self.primary.run(cmd)

        console.print("[dim]Inference complete. Pulling latest metrics to local machine...[/dim]")
        self.sync_experiments_down()


    ##----------------------------------------------------------------------------
    ## Deployment Methods
    ##----------------------------------------------------------------------------

    def _resolve_local_path(self, path_str: str) -> str:
        """
        Takes a path from config. If it's relative, anchors it to the Project Root.
        If it's absolute, leaves it alone.
        """
        path = Path(path_str)
        if path.is_absolute():
            return str(path)

        return str((self.project_root / path).resolve())

    def check_connectivity(self):
        """
        Verifies SSH access to all nodes.
        """
        console.rule("[bold]Checking Connectivity")
        try:
            # Run 'hostname' on all nodes in parallel
            results = self.group.run('hostname', hide=True)
            for connection, result in results.items():
                console.print(f"[green]✓ Connected to {connection.host}: {result.stdout.strip()}[/green]")
        except Exception as e:
            console.print(f"[red]✗ Connection failed: {e}[/red]")
            raise e

    def deploy_worker_code(self):
        """
        Syncs the local spt-worker to all worker nodes.
        """
        console.rule("[bold]Deploying ScalePT Workers")

        local_path = self.cfg.paths.local_worker_repo
        remote_path = self.cfg.paths.remote_worker_repo
        exclude_flags = "--exclude '.git' --exclude '__pycache__' --exclude '*.pyc' --exclude '.DS_Store'"

        console.print(f"[yellow]Syncing {local_path} -> {remote_path}...[/yellow]")

        for node in self.cfg.worker_nodes:
            ip = node.ip
            cmd = (
                f"rsync -az {exclude_flags} -e 'ssh -i {self.cfg.project.key_filename}' "
                f"{local_path}/ {self.user}@{ip}:{remote_path}"
            )

            os.system(cmd)
            console.print(f"[green]✓ Synced to {node.name}[/green]")

        console.print("[bold green]✓ Code deployment complete.[/bold green]")

    def deploy_dataset(self):
        """
        Transfers the dataset specified in the config to the Primary Node (NFS host).
        """
        console.rule("[bold]Syncing Dataset to NFS")

        local_data = os.path.abspath(self.cfg.paths.local_dataset)
        remote_data = f"{self.base_dir}/dataset"

        host = self.cfg.primary_node.ip

        console.print(f"[yellow]Target: {host}:{remote_data}[/yellow]")
        console.print("[dim]This may take a while if changes are large...[/dim]")

        # Ensure remote directory exists
        self.primary.run(f"mkdir -p {self.remote_dataset_dir}")
        self.primary.run(f"mkdir -p {self.remote_weights_dir}")
        self.primary.run(f"mkdir -p {self.remote_experiments_dir}")

        cmd = (
            f"rsync -avz --progress -e 'ssh -i {self.cfg.project.key_filename}' "
            f"{local_data}/ {self.user}@{host}:{remote_data}"
        )

        os.system(cmd)
        console.print("[bold green]✓ Dataset sync complete.[/bold green]")

    def setup_remote_environments(self):
        console.rule("[bold]Bootstrapping Remote Environments")

        # CUDA 12.1 check
        system_check = (
            "if [ ! -d '/usr/local/cuda-12.1' ]; then "
            "  echo 'CRITICAL ERROR: System CUDA 12.1 not found in /usr/local/cuda-12.1!'; "
            "  exit 1; "
            "fi"
        )

        # Env setup script
        setup_script = (
            f"{system_check}; "
            "source ~/miniconda3/etc/profile.d/conda.sh; "
            # Conda env
            "conda activate spt-worker || conda create -n spt-worker python=3.11 -y; "
            "conda activate spt-worker; "
            "conda env update -n spt-worker -f ~/spt-worker/environment.yml --prune; "
            # Pytorch setup
            "pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118; "
            "pip install spconv-cu118; "
            "pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html; "
            "pip install torch_geometric; "
        )

        console.print("[yellow]Installing CUDA + PyTorch 11.8...[/yellow]")

        try:
            self.group.run(setup_script, hide=True)
            results = self.group.run("ls -d ~/miniconda3/envs/spt-worker || echo 'MISSING'", hide=True)
            for conn, res in results.items():
                if "MISSING" in res.stdout:
                    console.print(f"[red]✗ Conda env missing on {conn.host}[/red]")
                    raise Exception(f"Conda env missing on {conn.host}")
                else:
                    console.print(f"[green]✓ Conda env active on {conn.host}[/green]")
            console.print("[green]✓ All environments ready.[/green]")
        except Exception as e:
            console.print(f"[bold red]✗ Installation failed![/bold red]")
            if hasattr(e, 'result'):
                for conn, outcome in e.result.items():
                    if isinstance(outcome, Exception):
                        console.print(f"[red]--- Failure Log for {conn.host} ---[/red]")
                        if hasattr(outcome, 'result'):
                            console.print(outcome.result.stderr)
            raise e

    def ensure_remote_dirs(self):
        # Only on the NFS Host
        remote_exp = self.cfg.paths.remote_experiments
        remote_data = self.cfg.paths.remote_dataset

        self.primary.run(f"mkdir -p {remote_exp} {remote_data}")

    def sync_experiments_down(self):
        """
        Downloads the experiment logs/metrics from NFS to local
        """
        console.rule("[bold]Syncing Experiments (Remote -> Local)")

        check = self.primary.run(f"[ -d {self.remote_experiments_dir} ]", warn=True, hide=True)
        if check.failed:
            console.print("[yellow]! No experiment folder found. Creating it now...[/yellow]")
            self.ensure_remote_dirs()
            console.print("[yellow]! Zero experiments available to sync.[/yellow]")
            return

        local_dir = os.path.join(self.project_root, "notebooks", "experiments")
        os.makedirs(local_dir, exist_ok=True)

        cmd = (
            f"rsync -az --exclude '*.pt' -e 'ssh -i {self.cfg.project.key_filename}' "
            f"{self.user}@{self.cfg.primary_node.ip}:{self.remote_experiments_dir}/ {local_dir}/"
        )

        import subprocess
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode == 0:
            console.print(f"[green]✓ Experiments synced locally to {local_dir}.[/green]")
        else:
            console.print(f"[red]Sync warning: {result.stderr}[/red]")


    def resume_distributed_training(self, experiment_folder_name: str):
        """
        Resumes a crashed or stopped training run from the latest checkpoint
        """
        console.rule(f"[bold]Resuming Distributed Training: {experiment_folder_name}")

        experiment_dir = f"{self.remote_experiments_dir}/{experiment_folder_name}"
        weights_dir = f"{experiment_dir}/weights"

        if self.primary.run(f"[ -d {experiment_dir} ]", warn=True, hide=True).failed:
            console.print(f"[red]Critical: Experiment directory not found: {experiment_dir}[/red]")
            return

        cmd_find_weights = f"ls {weights_dir}/model_weights_epoch_*.pt | sort -V | tail -n 1"
        res = self.primary.run(cmd_find_weights, warn=True, hide=True)
        latest_checkpoint = res.stdout.strip()

        if not latest_checkpoint:
            console.print(f"[red]Critical: No epoch checkpoints found in {weights_dir}[/red]")
            return

        console.print(f"[green]Found latest checkpoint:[/green] {latest_checkpoint}")

        # get the original config
        res_config = self.primary.run(f"cat {experiment_dir}/run_config.json", hide=True)
        import json
        try:
            config_data = json.loads(res_config.stdout)
            sampling_strategy = config_data["arguments"]["sampling_strategy"]
            sequences = " ".join(config_data["arguments"]["sequences"])
            epochs = config_data["arguments"]["epochs"]
        except Exception as e:
            console.print(f"[red]Failed to parse run_config.json: {e}[/red]")
            return

        import datetime
        resume_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        master_addr = self.cfg.primary_node.ip
        master_port = "29500"
        nnodes = self.cfg.args.nnodes_used
        nproc_per_node = self.cfg.args.nproc_per_node

        cmd_template = (
            "nohup bash -c '"
            "source ~/miniconda3/etc/profile.d/conda.sh && "
            "conda activate spt-worker && "
            "export PATH=/usr/local/cuda-12.1/bin:$PATH && "
            "export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH && "
            "cd ~/spt-worker && "
            "torchrun "
            f"--nproc_per_node={nproc_per_node} "
            f"--nnodes={nnodes} "
            "--node_rank={rank} "
            f"--master_addr={master_addr} "
            f"--master_port={master_port} "
            "-m spt_worker.train "
            f"--data_path {self.remote_dataset_dir} "
            f"--labels_path {self.remote_dataset_dir} "
            f"--output_dir {experiment_dir} "
            f"--sampling_strategy {sampling_strategy} "
            f"--sequences {sequences} "
            "--accumulation_steps 4 "
            "--num_workers 2 "
            f"--epochs {epochs} "
            f"--resume {latest_checkpoint} "
            f">> {experiment_dir}/logs/resume_{resume_timestamp}_rank{{rank}}.log 2>&1"
            "' > /dev/null 2>&1 &"
        )

        for i, node in enumerate(self.cfg.worker_nodes):
            rank = i
            node_ip = node.ip
            cmd = cmd_template.format(rank=rank)
            console.print(f"[dim]Launching Resume Rank {rank} on {node_ip}...[/dim]")

            from fabric import Connection
            conn = Connection(host=node_ip, user=self.user,
                              connect_kwargs={"key_filename": self.cfg.project.key_filename})
            conn.run(cmd, hide=True)

        console.print("[bold green]✓ Training resumed successfully![/bold green]")