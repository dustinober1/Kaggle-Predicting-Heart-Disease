# Kaggle Notebooks Collection

A collection of Jupyter notebooks exploring different Kaggle datasets. Each notebook is self-contained and focuses on a specific dataset and modeling workflow.

## Repository Structure

- Each notebook lives at the repo root (or in a dataset-named folder, if added later).
- Notebooks include their own data loading and evaluation steps.
- Outputs (figures or artifacts) are generated inside the notebook and are not committed unless explicitly needed.

## How To Run

1. Open a notebook in Jupyter or VS Code.
2. Ensure the dataset is available in the expected location.
   - Kaggle kernels: `/kaggle/input/...`
   - Local: update file paths in the notebook to match your download location.
3. Run the cells from top to bottom.

## Datasets

Each notebook documents its dataset source and assumptions near the top of the file. If a dataset path changes, update the discovery cell or the configured paths.

## Environment Notes

- Most notebooks assume Python 3.10+.
- Common dependencies: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.
- Some notebooks may require larger models (GPU) and extra packages (for example, `transformers`, `dspy-ai`).

## Contributing

If you add a new notebook:

- Include a short EDA or dataset validation section.
- Document key assumptions and results in markdown cells.
- Keep the notebook runnable top-to-bottom.
