# OULAD Week-4 Early Outcome Prediction

This repository contains a compact public snapshot of an OULAD early-warning project. The task is to predict favourable versus unfavourable student outcomes using only information available by week 4, with a focus on leakage-safe feature engineering and clear model evaluation.

## Included Files

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

## Notes

This public snapshot is intentionally smaller than the full coursework workspace. Supporting report-generation files and private submission documents were left out so the repository stays focused on the notebook, the main pipeline, and the minimal files needed to understand the project.
