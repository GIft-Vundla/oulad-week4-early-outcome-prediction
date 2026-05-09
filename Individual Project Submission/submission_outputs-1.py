from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from week4_pipeline import (
    RANDOM_STATE,
    PipelineConfig,
    build_logistic_pipeline,
    get_model_columns,
    normalize_model_frame,
    prepare_catboost_inputs,
    run_full_pipeline,
    split_dataset,
)


PALETTE = {
    "Unfavourable": "#d1495b",
    "Favourable": "#2a9d8f",
}


def format_metric(value: float) -> str:
    return f"{value:.4f}"


def save_methodology_block_diagram(config: PipelineConfig) -> Path:
    path = config.submission_plots_dir / "figure_1_methodology_block_diagram.png"
    fig, ax = plt.subplots(figsize=(19, 5.4))
    ax.set_xlim(0, 1.34)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box_specs = {
        "raw": {
            "x": 0.02,
            "y": 0.31,
            "w": 0.12,
            "h": 0.38,
            "label": "Raw OULAD CSVs\nstudentInfo\nstudentVle\nvle\nstudentRegistration\nassessments\nstudentAssessment",
            "facecolor": "#f7f7f7",
            "edgecolor": "#333333",
        },
        "filter": {
            "x": 0.19,
            "y": 0.31,
            "w": 0.15,
            "h": 0.38,
            "label": "Week-4 filtering\nkeep day 0-27 only\nremove exact VLE duplicates\nexclude leakage fields",
            "facecolor": "#f7f7f7",
            "edgecolor": "#333333",
        },
        "aggregate": {
            "x": 0.39,
            "y": 0.31,
            "w": 0.16,
            "h": 0.38,
            "label": "Feature aggregation\nstudent-module-\npresentation grain\nclick, registration,\nand assessment features",
            "facecolor": "#f7f7f7",
            "edgecolor": "#333333",
        },
        "feature": {
            "x": 0.60,
            "y": 0.31,
            "w": 0.13,
            "h": 0.38,
            "label": "Feature table\none row per\nstudent-module-\npresentation",
            "facecolor": "#f7f7f7",
            "edgecolor": "#333333",
        },
        "split": {
            "x": 0.78,
            "y": 0.31,
            "w": 0.13,
            "h": 0.38,
            "label": "Train / validation /\ntest split\n60 / 20 / 20\nrandom_state=42",
            "facecolor": "#f7f7f7",
            "edgecolor": "#333333",
        },
        "pipeline_1": {
            "x": 0.98,
            "y": 0.58,
            "w": 0.17,
            "h": 0.20,
            "label": "Pipeline 1\nColumnTransformer\n+ LogisticRegression",
            "facecolor": "#eef5ff",
            "edgecolor": "#4a6fa5",
        },
        "pipeline_2": {
            "x": 0.98,
            "y": 0.14,
            "w": 0.17,
            "h": 0.20,
            "label": "Pipeline 2\nCatBoost\n+ class weights\n+ sigmoid calibration",
            "facecolor": "#eef5ff",
            "edgecolor": "#4a6fa5",
        },
        "output": {
            "x": 1.22,
            "y": 0.34,
            "w": 0.10,
            "h": 0.32,
            "label": "Threshold selection,\nmetrics, figures,\nand summary tables",
            "facecolor": "#e9f5ef",
            "edgecolor": "#2a9d8f",
        },
    }

    for spec in box_specs.values():
        patch = FancyBboxPatch(
            (spec["x"], spec["y"]),
            spec["w"],
            spec["h"],
            boxstyle="round,pad=0.02,rounding_size=0.01",
            linewidth=1.2,
            facecolor=spec["facecolor"],
            edgecolor=spec["edgecolor"],
        )
        ax.add_patch(patch)
        ax.text(
            spec["x"] + spec["w"] / 2,
            spec["y"] + spec["h"] / 2,
            spec["label"],
            ha="center",
            va="center",
            fontsize=9.5,
        )

    def right_center(name: str) -> tuple[float, float]:
        spec = box_specs[name]
        return spec["x"] + spec["w"], spec["y"] + spec["h"] / 2

    def left_center(name: str) -> tuple[float, float]:
        spec = box_specs[name]
        return spec["x"], spec["y"] + spec["h"] / 2

    for start_name, end_name in [
        ("raw", "filter"),
        ("filter", "aggregate"),
        ("aggregate", "feature"),
        ("feature", "split"),
    ]:
        ax.add_patch(
            FancyArrowPatch(
                right_center(start_name),
                left_center(end_name),
                arrowstyle="->",
                mutation_scale=10,
                linewidth=1.5,
                color="#333333",
                shrinkA=6,
                shrinkB=6,
            )
        )

    junction = (0.94, 0.50)
    ax.add_patch(Circle(junction, radius=0.006, color="#333333"))
    merge_point = (1.18, 0.50)
    ax.add_patch(Circle(merge_point, radius=0.006, color="#333333"))

    ax.add_patch(
        FancyArrowPatch(
            right_center("split"),
            junction,
            arrowstyle="-",
            linewidth=1.5,
            color="#333333",
            shrinkA=6,
            shrinkB=0,
        )
    )
    for target_name in ["pipeline_1", "pipeline_2"]:
        ax.add_patch(
            FancyArrowPatch(
                junction,
                left_center(target_name),
                arrowstyle="->",
                mutation_scale=10,
                linewidth=1.5,
                color="#333333",
                shrinkA=0,
                shrinkB=6,
            )
        )
        ax.add_patch(
            FancyArrowPatch(
                right_center(target_name),
                merge_point,
                arrowstyle="->",
                mutation_scale=10,
                linewidth=1.5,
                color="#333333",
                shrinkA=6,
                shrinkB=0,
                connectionstyle=(
                    "angle3,angleA=0,angleB=90"
                    if target_name == "pipeline_1"
                    else "angle3,angleA=0,angleB=-90"
                ),
            )
        )
    ax.add_patch(
        FancyArrowPatch(
            merge_point,
            left_center("output"),
            arrowstyle="->",
            mutation_scale=10,
            linewidth=1.5,
            color="#333333",
            shrinkA=0,
            shrinkB=6,
        )
    )

    plt.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def save_positive_eda_figures(
    config: PipelineConfig, feature_table: pd.DataFrame
) -> dict[str, Path]:
    sns.set_theme(style="whitegrid")
    order = ["Unfavourable", "Favourable"]
    figure_specs = [
        (
            "figure_2_total_clicks_w4_by_target.png",
            "total_clicks_w4",
            "Week-4 total clicks by binary outcome",
            "total_clicks_w4",
        ),
        (
            "figure_3_active_days_w4_by_target.png",
            "active_days_w4",
            "Week-4 active days by binary outcome",
            "active_days_w4",
        ),
        (
            "figure_4_clicks_per_active_day_w4_by_target.png",
            "clicks_per_active_day_w4",
            "Week-4 clicks per active day by binary outcome",
            "clicks_per_active_day_w4",
        ),
        (
            "figure_5_weighted_submitted_by_w4_by_target.png",
            "weighted_submitted_by_w4",
            "Early assessment weight submitted by binary outcome",
            "weighted_submitted_by_w4",
        ),
    ]
    outputs: dict[str, Path] = {}
    for filename, column, title, ylabel in figure_specs:
        path = config.submission_plots_dir / filename
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(
            data=feature_table,
            x="binary_target",
            y=column,
            hue="binary_target",
            order=order,
            palette=PALETTE,
            showfliers=False,
            legend=False,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Binary target")
        ax.set_ylabel(ylabel)
        plt.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)
        outputs[column] = path
    return outputs


def save_confusion_matrix_figure(
    config: PipelineConfig, comparison_table: pd.DataFrame
) -> Path:
    path = config.submission_plots_dir / "figure_6_confusion_matrices.png"
    display_names = {
        "Logistic Regression": "Logistic Regression\nthreshold 0.5",
        "Logistic Regression (tuned threshold)": "Logistic Regression\nthreshold 0.3",
        "CatBoost": "CatBoost\nthreshold 0.5",
        "Calibrated CatBoost (tuned threshold)": "Calibrated CatBoost\nthreshold 0.3",
    }
    order = [
        "Logistic Regression",
        "Logistic Regression (tuned threshold)",
        "CatBoost",
        "Calibrated CatBoost (tuned threshold)",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, model_name in zip(axes.flat, order, strict=True):
        row = comparison_table.loc[comparison_table["model"].eq(model_name)].iloc[0]
        matrix = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".0f",
            cmap="Blues",
            cbar=False,
            ax=ax,
            square=True,
        )
        ax.set_title(display_names[model_name])
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_xticklabels(["Unfavourable", "Favourable"], rotation=20)
        ax.set_yticklabels(["Unfavourable", "Favourable"], rotation=0)
    plt.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def save_grouped_metric_comparison_figure(
    config: PipelineConfig, comparison_table: pd.DataFrame
) -> Path:
    path = config.submission_plots_dir / "figure_9_metric_comparison.png"
    display_names = {
        "Logistic Regression": "P1",
        "Logistic Regression (tuned threshold)": "P1-T",
        "CatBoost": "P2",
        "Calibrated CatBoost (tuned threshold)": "P2-T",
    }
    plot_df = comparison_table.copy()
    plot_df["configuration"] = plot_df["model"].map(display_names)
    plot_df = plot_df.melt(
        id_vars=["configuration"],
        value_vars=["accuracy", "recall", "f1", "roc_auc"],
        var_name="metric",
        value_name="value",
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=plot_df,
        x="metric",
        y="value",
        hue="configuration",
        ax=ax,
    )
    ax.set_title("Metric comparison across reported configurations")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def copy_existing_plot(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())
    return destination


def compute_outlier_count(series: pd.Series) -> int:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return 0
    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return int(((numeric < lower) | (numeric > upper)).sum())


def infer_feature_spec(feature: str, vle_duplicate_count: int) -> dict[str, Any]:
    demographic_specs = {
        "code_module": {
            "Source CSV(s)": "studentInfo.csv",
            "Original column(s) used": "code_module",
            "Data type": "cat",
            "Definition": "Module identifier.",
            "How computed": "as-is",
            "Week availability (2/4/6/8)": "2/4/6/8",
            "Leakage risk": "None",
            "Notes": "Static course identifier.",
            "Duplicate count": 0,
        },
        "code_presentation": {
            "Source CSV(s)": "studentInfo.csv",
            "Original column(s) used": "code_presentation",
            "Data type": "cat",
            "Definition": "Presentation identifier.",
            "How computed": "as-is",
            "Week availability (2/4/6/8)": "2/4/6/8",
            "Leakage risk": "None",
            "Notes": "Static presentation identifier.",
            "Duplicate count": 0,
        },
        "gender": {
            "Source CSV(s)": "studentInfo.csv",
            "Original column(s) used": "gender",
            "Data type": "cat",
            "Definition": "Student gender category.",
            "How computed": "as-is",
            "Week availability (2/4/6/8)": "2/4/6/8",
            "Leakage risk": "Low",
            "Notes": "Fairness-sensitive demographic field.",
            "Duplicate count": 0,
        },
        "region": {
            "Source CSV(s)": "studentInfo.csv",
            "Original column(s) used": "region",
            "Data type": "cat",
            "Definition": "Student region category.",
            "How computed": "as-is",
            "Week availability (2/4/6/8)": "2/4/6/8",
            "Leakage risk": "Low",
            "Notes": "Fairness-sensitive demographic field.",
            "Duplicate count": 0,
        },
        "highest_education": {
            "Source CSV(s)": "studentInfo.csv",
            "Original column(s) used": "highest_education",
            "Data type": "cat",
            "Definition": "Highest education band.",
            "How computed": "as-is",
            "Week availability (2/4/6/8)": "2/4/6/8",
            "Leakage risk": "Low",
            "Notes": "Static background feature.",
            "Duplicate count": 0,
        },
        "imd_band": {
            "Source CSV(s)": "studentInfo.csv",
            "Original column(s) used": "imd_band",
            "Data type": "cat",
            "Definition": "Socioeconomic deprivation band.",
            "How computed": "as-is",
            "Week availability (2/4/6/8)": "2/4/6/8",
            "Leakage risk": "Low",
            "Notes": "Fairness-sensitive socioeconomic proxy.",
            "Duplicate count": 0,
        },
        "age_band": {
            "Source CSV(s)": "studentInfo.csv",
            "Original column(s) used": "age_band",
            "Data type": "cat",
            "Definition": "Age category.",
            "How computed": "as-is",
            "Week availability (2/4/6/8)": "2/4/6/8",
            "Leakage risk": "Low",
            "Notes": "Static demographic band.",
            "Duplicate count": 0,
        },
        "num_of_prev_attempts": {
            "Source CSV(s)": "studentInfo.csv",
            "Original column(s) used": "num_of_prev_attempts",
            "Data type": "num",
            "Definition": "Number of prior attempts on the module.",
            "How computed": "as-is",
            "Week availability (2/4/6/8)": "2/4/6/8",
            "Leakage risk": "None",
            "Notes": "Static history feature.",
            "Duplicate count": 0,
        },
        "studied_credits": {
            "Source CSV(s)": "studentInfo.csv",
            "Original column(s) used": "studied_credits",
            "Data type": "num",
            "Definition": "Number of studied credits.",
            "How computed": "as-is",
            "Week availability (2/4/6/8)": "2/4/6/8",
            "Leakage risk": "None",
            "Notes": "Static workload feature.",
            "Duplicate count": 0,
        },
        "disability": {
            "Source CSV(s)": "studentInfo.csv",
            "Original column(s) used": "disability",
            "Data type": "cat",
            "Definition": "Disability disclosure flag.",
            "How computed": "as-is",
            "Week availability (2/4/6/8)": "2/4/6/8",
            "Leakage risk": "Low",
            "Notes": "Fairness-sensitive field.",
            "Duplicate count": 0,
        },
    }
    registration_specs = {
        "date_registration": {
            "Source CSV(s)": "studentRegistration.csv",
            "Original column(s) used": "date_registration",
            "Data type": "num",
            "Definition": "Registration day relative to presentation start.",
            "How computed": "as-is",
            "Week availability (2/4/6/8)": "2/4/6/8",
            "Leakage risk": "None",
            "Notes": "Known before or at course start.",
            "Duplicate count": 0,
        },
        "registered_before_start_days": {
            "Source CSV(s)": "studentRegistration.csv",
            "Original column(s) used": "date_registration",
            "Data type": "num",
            "Definition": "Days registered before day 0.",
            "How computed": "max(0, -date_registration)",
            "Week availability (2/4/6/8)": "2/4/6/8",
            "Leakage risk": "None",
            "Notes": "Derived from registration timing only.",
            "Duplicate count": 0,
        },
        "late_registration_flag": {
            "Source CSV(s)": "studentRegistration.csv",
            "Original column(s) used": "date_registration",
            "Data type": "bool",
            "Definition": "Whether registration happened after day 0.",
            "How computed": "1 if date_registration > 0 else 0",
            "Week availability (2/4/6/8)": "2/4/6/8",
            "Leakage risk": "None",
            "Notes": "Binary timing flag.",
            "Duplicate count": 0,
        },
    }
    click_specs = {
        "total_clicks_w4": {
            "Source CSV(s)": "studentVle.csv",
            "Original column(s) used": "date, sum_click",
            "Data type": "num",
            "Definition": "Total VLE clicks observed by week 4.",
            "How computed": "sum(sum_click) where 0 <= date <= 27",
            "Week availability (2/4/6/8)": "4/6/8",
            "Leakage risk": "Low",
            "Notes": "Week-4 aggregate after duplicate removal.",
            "Duplicate count": vle_duplicate_count,
        },
        "first_activity_day_w4": {
            "Source CSV(s)": "studentVle.csv",
            "Original column(s) used": "date",
            "Data type": "num",
            "Definition": "First valid activity day in the week-4 window.",
            "How computed": "min(date) where 0 <= date <= 27",
            "Week availability (2/4/6/8)": "4/6/8",
            "Leakage risk": "Low",
            "Notes": "Missing for inactive students.",
            "Duplicate count": vle_duplicate_count,
        },
        "active_days_w4": {
            "Source CSV(s)": "studentVle.csv",
            "Original column(s) used": "date",
            "Data type": "num",
            "Definition": "Number of distinct active click days by week 4.",
            "How computed": "nunique(date) where 0 <= date <= 27",
            "Week availability (2/4/6/8)": "4/6/8",
            "Leakage risk": "Low",
            "Notes": "Zero-filled for inactive students.",
            "Duplicate count": vle_duplicate_count,
        },
        "unique_sites_w4": {
            "Source CSV(s)": "studentVle.csv",
            "Original column(s) used": "id_site",
            "Data type": "num",
            "Definition": "Number of distinct VLE sites visited by week 4.",
            "How computed": "nunique(id_site) where 0 <= date <= 27",
            "Week availability (2/4/6/8)": "4/6/8",
            "Leakage risk": "Low",
            "Notes": "Site breadth feature.",
            "Duplicate count": vle_duplicate_count,
        },
        "clicks_per_active_day_w4": {
            "Source CSV(s)": "studentVle.csv",
            "Original column(s) used": "date, sum_click",
            "Data type": "num",
            "Definition": "Average clicks per active day by week 4.",
            "How computed": "total_clicks_w4 / active_days_w4",
            "Week availability (2/4/6/8)": "4/6/8",
            "Leakage risk": "Low",
            "Notes": "Intensity measure.",
            "Duplicate count": vle_duplicate_count,
        },
        "days_since_last_activity_w4": {
            "Source CSV(s)": "studentVle.csv",
            "Original column(s) used": "date",
            "Data type": "num",
            "Definition": "Recency of the last valid week-4 activity.",
            "How computed": "27 - max(date) where 0 <= date <= 27",
            "Week availability (2/4/6/8)": "4/6/8",
            "Leakage risk": "Low",
            "Notes": "Missing for inactive students.",
            "Duplicate count": vle_duplicate_count,
        },
        "recommended_clicks_w4": {
            "Source CSV(s)": "studentVle.csv, vle.csv",
            "Original column(s) used": "date, sum_click, recommended",
            "Data type": "num",
            "Definition": "Clicks on resources flagged as recommended.",
            "How computed": "sum(sum_click) on recommended resources where 0 <= date <= 27",
            "Week availability (2/4/6/8)": "4/6/8",
            "Leakage risk": "Low",
            "Notes": "The provided `vle.csv` has no `recommended` column; values are reported as 0.",
            "Duplicate count": vle_duplicate_count,
        },
        "recommended_click_ratio_w4": {
            "Source CSV(s)": "studentVle.csv, vle.csv",
            "Original column(s) used": "date, sum_click, recommended",
            "Data type": "num",
            "Definition": "Share of total week-4 clicks on recommended resources.",
            "How computed": "recommended_clicks_w4 / total_clicks_w4",
            "Week availability (2/4/6/8)": "4/6/8",
            "Leakage risk": "Low",
            "Notes": "The provided `vle.csv` has no `recommended` column; values are reported as 0.",
            "Duplicate count": vle_duplicate_count,
        },
    }
    assessment_specs = {
        "early_assessment_due_count": {
            "Source CSV(s)": "assessments.csv",
            "Original column(s) used": "id_assessment, date",
            "Data type": "num",
            "Definition": "Number of assessments due by day 27 for the module presentation.",
            "How computed": "count distinct id_assessment where 0 <= date <= 27",
            "Week availability (2/4/6/8)": "4/6/8",
            "Leakage risk": "Low",
            "Notes": "Module-presentation level feature.",
            "Duplicate count": 0,
        },
        "weighted_due_by_w4": {
            "Source CSV(s)": "assessments.csv",
            "Original column(s) used": "date, weight",
            "Data type": "num",
            "Definition": "Total assessment weight due by day 27.",
            "How computed": "sum(weight) where 0 <= date <= 27",
            "Week availability (2/4/6/8)": "4/6/8",
            "Leakage risk": "Low",
            "Notes": "Module-presentation level feature.",
            "Duplicate count": 0,
        },
        "early_assessment_submitted_count": {
            "Source CSV(s)": "assessments.csv, studentAssessment.csv",
            "Original column(s) used": "id_assessment, id_student, date_submitted",
            "Data type": "num",
            "Definition": "Number of early assessments submitted by day 27.",
            "How computed": "count distinct id_assessment after joining early assessments and submissions",
            "Week availability (2/4/6/8)": "4/6/8",
            "Leakage risk": "Low",
            "Notes": "Submission behavior feature.",
            "Duplicate count": 0,
        },
        "on_time_submission_count": {
            "Source CSV(s)": "assessments.csv, studentAssessment.csv",
            "Original column(s) used": "date, date_submitted",
            "Data type": "num",
            "Definition": "Count of early assessments submitted on or before the due date.",
            "How computed": "sum(1 where date_submitted - due_date <= 0)",
            "Week availability (2/4/6/8)": "4/6/8",
            "Leakage risk": "Low",
            "Notes": "Submission punctuality feature.",
            "Duplicate count": 0,
        },
        "late_submission_count": {
            "Source CSV(s)": "assessments.csv, studentAssessment.csv",
            "Original column(s) used": "date, date_submitted",
            "Data type": "num",
            "Definition": "Count of early assessments submitted late but by day 27.",
            "How computed": "sum(1 where date_submitted - due_date > 0)",
            "Week availability (2/4/6/8)": "4/6/8",
            "Leakage risk": "Low",
            "Notes": "Submission punctuality feature.",
            "Duplicate count": 0,
        },
        "mean_submission_delay_days": {
            "Source CSV(s)": "assessments.csv, studentAssessment.csv",
            "Original column(s) used": "date, date_submitted",
            "Data type": "num",
            "Definition": "Average submission delay for early assessments.",
            "How computed": "mean(date_submitted - due_date)",
            "Week availability (2/4/6/8)": "4/6/8",
            "Leakage risk": "Low",
            "Notes": "Zero-filled when no early submissions exist.",
            "Duplicate count": 0,
        },
        "weighted_submitted_by_w4": {
            "Source CSV(s)": "assessments.csv, studentAssessment.csv",
            "Original column(s) used": "weight, date_submitted",
            "Data type": "num",
            "Definition": "Total assessment weight submitted by day 27.",
            "How computed": "sum(weight) for early assessments submitted by day 27",
            "Week availability (2/4/6/8)": "4/6/8",
            "Leakage risk": "Low",
            "Notes": "Strong early coursework signal.",
            "Duplicate count": 0,
        },
        "submitted_any_early_assessment": {
            "Source CSV(s)": "assessments.csv, studentAssessment.csv",
            "Original column(s) used": "id_assessment, date_submitted",
            "Data type": "bool",
            "Definition": "Whether any early assessment was submitted by day 27.",
            "How computed": "1 if early_assessment_submitted_count > 0 else 0",
            "Week availability (2/4/6/8)": "4/6/8",
            "Leakage risk": "Low",
            "Notes": "Binary early-engagement flag.",
            "Duplicate count": 0,
        },
    }

    if feature in demographic_specs:
        return demographic_specs[feature]
    if feature in registration_specs:
        return registration_specs[feature]
    if feature in click_specs:
        return click_specs[feature]
    if feature in assessment_specs:
        return assessment_specs[feature]
    if feature.endswith("_clicks_w4"):
        activity_name = feature.replace("_clicks_w4", "")
        label = activity_name.replace("_", " ")
        return {
            "Source CSV(s)": "studentVle.csv, vle.csv",
            "Original column(s) used": "date, sum_click, activity_type",
            "Data type": "num",
            "Definition": f"Week-4 clicks on {label} resources.",
            "How computed": f"sum(sum_click) where activity_type = '{activity_name}' and 0 <= date <= 27",
            "Week availability (2/4/6/8)": "4/6/8",
            "Leakage risk": "Low",
            "Notes": "Activity-type engagement feature.",
            "Duplicate count": vle_duplicate_count,
        }
    raise KeyError(f"No feature specification defined for {feature}")


def build_feature_catalog(
    feature_table: pd.DataFrame, row_counts: dict[str, Any]
) -> pd.DataFrame:
    feature_columns, _, _ = get_model_columns(feature_table)
    vle_duplicate_count = int(row_counts.get("studentVle_week4_exact_duplicates_removed", 0))
    rows = []
    for feature in feature_columns:
        spec = infer_feature_spec(feature, vle_duplicate_count)
        outlier_value: str | int
        if spec["Data type"] in {"cat", "bool"}:
            outlier_value = "N/A"
        else:
            outlier_value = compute_outlier_count(feature_table[feature])
        rows.append(
            {
                "Feature": feature,
                "Source CSV(s)": spec["Source CSV(s)"],
                "Original column(s) used": spec["Original column(s) used"],
                "Data type": spec["Data type"],
                "Definition": spec["Definition"],
                "How computed": spec["How computed"],
                "Week availability (2/4/6/8)": spec["Week availability (2/4/6/8)"],
                "Missing count": int(feature_table[feature].isna().sum()),
                "Outliers count": outlier_value,
                "Duplicate count": spec["Duplicate count"],
                "Leakage risk": spec["Leakage risk"],
                "Notes": spec["Notes"],
            }
        )
    return pd.DataFrame(rows)


def build_pipeline_comparison_table(comparison_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    id_map = {
        "Logistic Regression": "P1",
        "Logistic Regression (tuned threshold)": "P1-T",
        "CatBoost": "P2",
        "Calibrated CatBoost (tuned threshold)": "P2-T",
    }
    for row in comparison_table.itertuples(index=False):
        is_logistic = "Logistic Regression" in row.model
        is_calibrated = "Calibrated" in row.model
        rows.append(
            {
                "Pipeline ID": id_map[row.model],
                "Feature set": "All week-4 features",
                "Encoding": (
                    "Most-frequent imputation + one-hot categorical encoding"
                    if is_logistic
                    else "Native categorical handling with explicit Missing category"
                ),
                "Scaling": "StandardScaler on numeric columns only" if is_logistic else "None",
                "Model": (
                    "LogisticRegression(max_iter=1000)"
                    if is_logistic
                    else (
                        "Calibrated CatBoostClassifier (sigmoid)"
                        if is_calibrated
                        else "CatBoostClassifier"
                    )
                ),
                "Evaluation method": "Split 60/20/20 with threshold selection on validation; 5-fold CV summary on training",
                "Hyperparameters": (
                    "max_iter=1000, random_state=42"
                    if is_logistic
                    else "loss_function=Logloss, eval_metric=AUC, depth=6, learning_rate=0.05, iterations=500, auto_class_weights=Balanced, random_state=42"
                ),
                "Seed": RANDOM_STATE,
                "Chosen threshold": row.threshold,
                "Accuracy": format_metric(row.accuracy),
                "Precision": format_metric(row.precision),
                "Recall": format_metric(row.recall),
                "Specificity": format_metric(row.specificity),
                "FPR": format_metric(row.fpr),
                "F1": format_metric(row.f1),
                "Log loss": format_metric(row.log_loss),
                "AUC": format_metric(row.roc_auc),
            }
        )
    return pd.DataFrame(rows)


def run_training_cross_validation(
    feature_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train, _, _, y_train, _, _ = split_dataset(feature_table)
    _, categorical_columns, numeric_columns = get_model_columns(feature_table)
    X_train = normalize_model_frame(X_train, categorical_columns, numeric_columns)
    y_train = y_train.reset_index(drop=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_rows: list[dict[str, Any]] = []

    for fold_index, (train_idx, valid_idx) in enumerate(skf.split(X_train, y_train), start=1):
        X_fold_train = X_train.iloc[train_idx].reset_index(drop=True)
        X_fold_valid = X_train.iloc[valid_idx].reset_index(drop=True)
        y_fold_train = y_train.iloc[train_idx].reset_index(drop=True)
        y_fold_valid = y_train.iloc[valid_idx].reset_index(drop=True)

        logistic = build_logistic_pipeline(categorical_columns, numeric_columns)
        logistic.fit(X_fold_train, y_fold_train)
        logistic_valid_proba = logistic.predict_proba(X_fold_valid)[:, 1]
        fold_rows.append(
            {
                "model": "Logistic Regression",
                "fold": fold_index,
                "roc_auc": roc_auc_score(y_fold_valid, logistic_valid_proba),
                "log_loss": log_loss(y_fold_valid, logistic_valid_proba, labels=[0, 1]),
            }
        )

        X_fold_train_cat = prepare_catboost_inputs(X_fold_train, categorical_columns)
        X_fold_valid_cat = prepare_catboost_inputs(X_fold_valid, categorical_columns)
        catboost = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            depth=6,
            learning_rate=0.05,
            iterations=500,
            auto_class_weights="Balanced",
            random_state=RANDOM_STATE,
            verbose=False,
        )
        catboost.fit(
            X_fold_train_cat,
            y_fold_train,
            cat_features=categorical_columns,
            eval_set=(X_fold_valid_cat, y_fold_valid),
            use_best_model=True,
            verbose=False,
        )
        catboost_valid_proba = catboost.predict_proba(X_fold_valid_cat)[:, 1]
        fold_rows.append(
            {
                "model": "CatBoost",
                "fold": fold_index,
                "roc_auc": roc_auc_score(y_fold_valid, catboost_valid_proba),
                "log_loss": log_loss(y_fold_valid, catboost_valid_proba, labels=[0, 1]),
            }
        )

    fold_scores = pd.DataFrame(fold_rows)
    summary = (
        fold_scores.groupby("model", as_index=False)
        .agg(
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            log_loss_mean=("log_loss", "mean"),
            log_loss_std=("log_loss", "std"),
        )
        .reset_index(drop=True)
    )
    summary["roc_auc_mean"] = summary["roc_auc_mean"].map(format_metric)
    summary["roc_auc_std"] = summary["roc_auc_std"].map(format_metric)
    summary["log_loss_mean"] = summary["log_loss_mean"].map(format_metric)
    summary["log_loss_std"] = summary["log_loss_std"].map(format_metric)
    return fold_scores, summary


def build_submission_tables(
    feature_table: pd.DataFrame,
    row_counts: dict[str, Any],
    comparison_table: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    class_balance = (
        feature_table.groupby("binary_target", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("binary_target")
        .reset_index(drop=True)
    )
    class_balance["Percentage"] = (
        class_balance["count"] / class_balance["count"].sum() * 100
    ).map(lambda value: f"{value:.1f}%")
    class_balance["Original OULAD values"] = class_balance["binary_target"].map(
        {
            "Favourable": "Pass, Distinction",
            "Unfavourable": "Fail, Withdrawn",
        }
    )
    class_balance = class_balance.rename(
        columns={"binary_target": "Binary target", "count": "Count"}
    )[["Binary target", "Original OULAD values", "Count", "Percentage"]]

    row_count_table = pd.DataFrame(
        [
            ("studentInfo.csv", row_counts.get("studentInfo", 0)),
            ("studentRegistration.csv", row_counts.get("studentRegistration", 0)),
            ("vle.csv", row_counts.get("vle", 0)),
            ("studentVle.csv", row_counts.get("studentVle", 0)),
            ("studentVle rows within day 0-27", row_counts.get("studentVle_week4_events", 0)),
            (
                "Exact duplicate studentVle rows removed before aggregation",
                row_counts.get("studentVle_week4_exact_duplicates_removed", 0),
            ),
            ("assessments.csv", row_counts.get("assessments", 0)),
            ("Assessments due by day 27", row_counts.get("assessments_due_by_week4", 0)),
            ("studentAssessment.csv", row_counts.get("studentAssessment", 0)),
            (
                "Assessment submissions by day 27",
                row_counts.get("studentAssessment_submitted_by_week4", 0),
            ),
        ],
        columns=["Table or derived subset", "Rows"],
    )

    missing_summary = (
        feature_table.isna()
        .sum()
        .sort_values(ascending=False)
        .rename("Missing count")
        .reset_index()
        .rename(columns={"index": "Column"})
        .loc[lambda df: df["Missing count"].gt(0)]
        .rename(columns={"column": "Column", "missing_count": "Missing count"})
        .reset_index(drop=True)
    )

    feature_catalog = build_feature_catalog(feature_table, row_counts)
    pipeline_table = build_pipeline_comparison_table(comparison_table)
    cv_folds, cv_summary = run_training_cross_validation(feature_table)

    return {
        "class_balance": class_balance,
        "row_counts": row_count_table,
        "missing_summary": missing_summary,
        "feature_catalog": feature_catalog,
        "pipeline_comparison": pipeline_table,
        "cv_summary": cv_summary,
        "cv_folds": cv_folds,
    }


def save_submission_tables(
    config: PipelineConfig,
    feature_table: pd.DataFrame,
    row_counts: dict[str, Any],
    model_outputs: dict[str, Any],
    table_outputs: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Path]:
    if table_outputs is None:
        table_outputs = build_submission_tables(
            feature_table,
            row_counts,
            model_outputs["comparison_table"].copy(),
        )

    output_paths: dict[str, Path] = {}
    for name, df in table_outputs.items():
        path = config.submission_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        output_paths[name] = path
    return output_paths


def build_submission_outputs(
    project_root: Path | str,
    force_rebuild: bool = False,
    save_tables: bool = True,
) -> dict[str, Any]:
    config = PipelineConfig.from_project_root(project_root)
    results = run_full_pipeline(project_root, force_rebuild=force_rebuild)
    feature_table = results["feature_table"]
    row_counts = results["row_counts"]

    methodology_path = save_methodology_block_diagram(config)
    eda_paths = save_positive_eda_figures(config, feature_table)
    confusion_path = save_confusion_matrix_figure(config, results["comparison_table"])
    metric_path = save_grouped_metric_comparison_figure(config, results["comparison_table"])
    roc_copy = copy_existing_plot(
        results["roc_plot_path"],
        config.submission_plots_dir / "figure_7_roc_curves.png",
    )
    threshold_copy = copy_existing_plot(
        results["threshold_plot_path"],
        config.submission_plots_dir / "figure_8_threshold_sweep.png",
    )

    table_data = build_submission_tables(
        feature_table,
        row_counts,
        results["comparison_table"].copy(),
    )
    table_paths = (
        save_submission_tables(
            config,
            feature_table,
            row_counts,
            results,
            table_outputs=table_data,
        )
        if save_tables
        else {}
    )

    figures = {
        "figure_1": methodology_path,
        "figure_2": eda_paths["total_clicks_w4"],
        "figure_3": eda_paths["active_days_w4"],
        "figure_4": eda_paths["clicks_per_active_day_w4"],
        "figure_5": eda_paths["weighted_submitted_by_w4"],
        "figure_6": confusion_path,
        "figure_7": roc_copy,
        "figure_8": threshold_copy,
        "figure_9": metric_path,
    }

    return {
        "submission_dir": config.submission_dir,
        "submission_plots_dir": config.submission_plots_dir,
        "feature_table_path": results["feature_table_path"],
        "feature_table": feature_table,
        "metrics_summary_path": results["metrics_summary_path"],
        "threshold_sweep_path": results["threshold_sweep_path"],
        "figures": figures,
        "tables": table_paths,
        "table_data": table_data,
        "results": results,
    }


def main() -> None:
    project_root = Path(__file__).resolve().parent
    outputs = build_submission_outputs(project_root, force_rebuild=False)
    summary = {
        "submission_dir": str(outputs["submission_dir"]),
        "feature_table_path": str(outputs["feature_table_path"]),
        "metrics_summary_path": str(outputs["metrics_summary_path"]),
        "threshold_sweep_path": str(outputs["threshold_sweep_path"]),
        "feature_catalog_path": str(outputs["tables"]["feature_catalog"]),
        "pipeline_comparison_path": str(outputs["tables"]["pipeline_comparison"]),
        "methodology_figure_path": str(outputs["figures"]["figure_1"]),
    }
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()
