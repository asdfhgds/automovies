"""Run the real GPU profile in Colab after scripts/colab_setup.sh."""
import os
import subprocess
import sys

os.environ.setdefault("STUDIO_PROFILE", "colab-gpu")
os.environ.setdefault("DIRECTOR_PROVIDER", "qwen")
os.environ.setdefault("SCRIPT_PROVIDER", "qwen")
os.environ.setdefault("DIRECTOR_MODEL", "Qwen/Qwen3-7B-A0.5B")

if __name__ == "__main__":
    subprocess.run([sys.executable, "src/main.py", "doctor"], check=True)
    print("Set a source with `python src/main.py init ...`, then run `python src/main.py run --project-id ...`.")
