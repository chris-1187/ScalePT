import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# Configuration
source_path = SCRIPT_DIR / "data" / "kitti"
destination = "cmunz@141.23.40.76:/mnt/nfs_share/"

# Check if source directory exists
if not source_path.is_dir():
    print(f"Error: Source path not found: {source_path}")
    sys.exit(1)

# Build the command
rsync_command = [
    "rsync",
    "-avP",
    str(source_path),
    destination
]

print(f"Starting rsync transfer...")
print(f"Source: {source_path}")
print(f"Destination: {destination}")
print(f"COMMAND: {' '.join(rsync_command)}\n")

try:
    with subprocess.Popen(rsync_command,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          text=True,
                          bufsize=1) as process:

        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                print(line, end='')

        process.wait()

        if process.returncode == 0:
            print(f"\n--- rsync COMPLETED successfully! ---")
        else:
            print(f"\n--- rsync FAILED with return code {process.returncode} ---")
            if process.stderr:
                stderr_output = process.stderr.read()
                print("\nSTDERR:")
                print(stderr_output)

except FileNotFoundError:
    print("\n--- ERROR ---")
    print("Command 'rsync' not found.")
    sys.exit(1)
except Exception as e:
    print(f"\n--- An unexpected error occurred ---")
    print(e)