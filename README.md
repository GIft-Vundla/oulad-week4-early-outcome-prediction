# OULAD Week-4 Early Outcome Prediction

This repository presents a week-4 early-warning workflow built on the Open University Learning Analytics Dataset (OULAD). The task is to predict favourable versus unfavourable student outcomes using only information available by week 4, with a focus on leakage-safe feature engineering and clear model evaluation.

## Included Files

- `OULAD_Individual_2550723.pdf`: final written report with figures, tables, and discussion
- `OULAD_Individual_2550723.ipynb`: final named notebook used for the project submission
- `week4_pipeline.py`: main feature-engineering, training, and evaluation workflow
- `requirements.txt`: Python dependencies required to run the pipeline locally

## Project Summary

- target definition: favourable (`Pass` or `Distinction`) versus unfavourable (`Fail` or `Withdrawn`)
- time window: only data available by day 27 is used for the modeled feature set
- modeling approach: a logistic-regression baseline compared with a CatBoost pipeline
- evaluation: confusion matrices, ROC-AUC, log loss, threshold analysis, and standard classification metrics

## Running The Workflow

Install the dependencies with:

```powershell
python -m pip install -r requirements.txt
```

Run the main pipeline with:

```powershell
python week4_pipeline.py
```

## Data

The raw OULAD CSV files are not included in this public repository. To rerun the workflow locally, place the dataset files in the project root:

- `assessments.csv`
- `courses.csv`
- `studentAssessment.csv`
- `studentInfo.csv`
- `studentRegistration.csv`
- `studentVle.csv`
- `vle.csv`

## Repository Focus

This repository is centered on the final report, the final notebook, and the core pipeline implementation.

- `OULAD_Individual_2550723.pdf` provides the written report and the final presented results.
- `OULAD_Individual_2550723.ipynb` provides the final project notebook.
- `week4_pipeline.py` contains the main feature-engineering, training, and evaluation workflow.
- `requirements.txt` captures the Python environment needed to run the project locally.

## Notes

The report, notebook, and pipeline are intended to be read together:

- the report summarises the methodology, figures, tables, and conclusions
- the notebook shows the final analysis workflow in notebook form
- the pipeline script contains the main reproducible implementation
