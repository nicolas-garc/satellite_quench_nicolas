# satellite_quench

Research notebooks for satellite-galaxy quenching analyses using SAGA, ELVES,
and TNG data.

## Environment setup

Create a dedicated Python environment for this repository so its dependencies
remain separate from other research projects:

```bash
cd ~/satellite_quench_nicolas
python3 -m venv ~/.venvs/satellite-quench
source ~/.venvs/satellite-quench/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user \
  --name satellite-quench \
  --display-name "Python (satellite-quench)"
```

Select `Python (satellite-quench)` when opening this repository's notebooks.
After pulling changes to the dependency file, refresh the environment with:

```bash
source ~/.venvs/satellite-quench/bin/activate
cd ~/satellite_quench_nicolas
python -m pip install -r requirements.txt
```

The exact clone path may differ between machines. Adjust the `cd` command, but
do not share this virtual environment with unrelated repositories.

## Running the notebooks

The notebooks currently load data using paths relative to the repository root.
Start Jupyter from the clone's top-level directory so those paths resolve
consistently:

```bash
cd ~/satellite_quench_nicolas
jupyter lab
```

`requirements.txt` lists direct dependencies without exact version pins so it
can be installed across the existing Binder and local environments. A tested,
platform-specific lock file can be added later when the supported Python
version and execution platforms are fixed.
