# Problem Set 3: Claims modelling exercise

```
mamba env create
conda activate ps3

pre-commit install
pip install --no-build-isolation -e .
```

With Pixi, install the local packages into its Python environment once:

```bash
pixi run python -m pip install --no-build-isolation --no-deps -e .
```

Run the complete PS3 and PS4 analysis from the repository root:

```bash
pixi run python -m analyses.ps3_script
```

The editable installation also lets you run `analyses/ps3_script.py` directly or
execute its `# %%` cells in order in an IDE using the Pixi Python interpreter.
The script loads and splits the data once, fits the benchmark GLM, spline GLM,
and unconstrained LightGBM, then continues through the PS4 constrained LightGBM,
learning curve, and evaluation. Plots appear along the way; close each plot
window to continue when running the whole script with an interactive backend.
