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
        config_path = config_dir / "cluster_config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Critical: Config not found at {config_path}")

        self.cfg = OmegaConf.load(config_path)
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

        console.print(f"[dim]Syncing {local_path} -> {remote_path}...[/dim]")

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
        remote_data = self.cfg.paths.remote_dataset
        host = self.cfg.primary_node.ip

        console.print(f"[yellow]Target: {host}:{remote_data}[/yellow]")
        console.print("[dim]This may take a while if changes are large...[/dim]")

        # Ensure remote directory exists
        self.primary.run(f"mkdir -p {remote_data}")

        cmd = (
            f"rsync -avz --progress -e 'ssh -i {self.cfg.project.key_filename}' "
            f"{local_data}/ {self.user}@{host}:{remote_data}"
        )

        os.system(cmd)
        console.print("[bold green]✓ Dataset sync complete.[/bold green]")

    def verify_remote_environment(self):
        """
        Checks if the conda environment is active/exists on workers.
        """
        console.rule("[bold]Verifying Environments")

        results = self.group.run("ls -d ~/miniconda3/envs/spt-worker || echo 'MISSING'", hide=True)

        for conn, res in results.items():
            if "MISSING" in res.stdout:
                console.print(f"[red]✗ Conda env missing on {conn.host}[/red]")
            else:
                console.print(f"[green]✓ Conda env found on {conn.host}[/green]")

    def install_remote_environments(self):
        """
        Installs conda environments if missing.
        """
        console.rule("[bold]Bootstrapping Remote Environments")

        setup_script = (
            "export PATH=$HOME/miniconda3/bin:$PATH; "
            "cd ~/spt-worker; "
            "if ! command -v conda &> /dev/null; then "
            "   echo 'CRITICAL: Conda not found in ~/miniconda3/bin'; exit 127; "
            "fi; "
            "conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main > /dev/null 2>&1 || true; "
            "conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r > /dev/null 2>&1 || true; "
            "if conda env list | grep -q 'spt-worker'; then "
            "   echo '>>> Environment exists. Updating...'; "
            "   conda env update -f environment.yml --prune; "
            "else "
            "   echo '>>> Environment missing. Creating...'; "
            "   conda env create -f environment.yml; "
            "fi"
        )

        console.print("[yellow]Installing/Updating Conda environments on all nodes...[/yellow]")
        console.print("[dim](This runs in parallel and may take a few minutes if creating new envs)[/dim]")

        try:
            # Run on all nodes
            self.group.run(setup_script, hide=True)
            console.print("[green]✓ All environments ready.[/green]")

        except Exception as e:
            console.print(f"[bold red]✗ Installation failed![/bold red]")
            if hasattr(e, 'result'):
                for conn, outcome in e.result.items():
                    if isinstance(outcome, Exception):
                        console.print(f"[red]--- Failure Log for {conn.host} ---[/red]")
                        if hasattr(outcome, 'result'):
                            console.print(outcome.result.stderr)
                        else:
                            console.print(str(outcome))
            raise e

    def ensure_remote_dirs(self):
        # Only on the NFS Host
        remote_exp = self.cfg.paths.remote_experiments
        remote_data = self.cfg.paths.remote_dataset

        self.primary.run(f"mkdir -p {remote_exp} {remote_data}")

    def sync_experiments_down(self):
        """
        Downloads the experiment logs/metrics from NFS to Local.
        """
        console.rule("[bold]Syncing Experiments (Remote -> Local)")

        check = self.primary.run(f"[ -d {self.cfg.paths.remote_experiments} ]", warn=True, hide=True)
        if check.failed:
            console.print("[yellow]! No experiment folder found. Creating it now...[/yellow]")
            self.ensure_remote_dirs()
            console.print("[yellow]! Zero experiments available to sync.[/yellow]")
            return

        local_dir = os.path.abspath(self.cfg.paths.local_experiments)
        os.makedirs(local_dir, exist_ok=True)

        cmd = (
            f"rsync -az --exclude '*.pt' -e 'ssh -i {self.cfg.project.key_filename}' "
            f"{self.user}@{self.cfg.primary_node.ip}:{self.cfg.paths.remote_experiments}/ {local_dir}/"
        )

        import subprocess
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode == 0:
            console.print("[green]✓ Experiments synced locally.[/green]")
        else:
            console.print(f"[red]Sync warning: {result.stderr}[/red]")