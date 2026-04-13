from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from pypdf import PdfReader
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


WEEK4_START_DAY = 0
WEEK4_END_DAY = 27
RANDOM_STATE = 42
EXPECTED_ACTIVITY_TYPES = [
    "forumng",
    "homepage",
    "oucontent",
    "quiz",
    "resource",
    "subpage",
    "url",
    "page",
]
CATEGORICAL_MODEL_COLUMNS = [
    "code_module",
    "code_presentation",
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "disability",
]

PAPER_CATALOG = [
    {
        "section": "Background and Problem Framing",
        "report_section": "Introduction / related work",
        "pdf_path": Path(
            "Background & Problem Framing (OULAD, Early Prediction, Binary Outcomes)"
        )
        / "953-Article Text-7982-10223-10-20251006.pdf",
        "title": "Early Prediction of At Risk Students Using Minimal Data: A Machine Learning Framework for Higher Education",
        "reference_hint": "Hamsiah, Adiyati, and Subekti (2025)",
        "venue": "Digitus: Journal of Computer Science Applications",
        "year": 2025,
        "selected_claim": "Week-4 OULAD features combined with CatBoost and logistic regression can support early risk prediction before major assessments.",
    },
    {
        "section": "Week-4 Clickstream Features",
        "report_section": "Feature engineering",
        "pdf_path": Path("Week 4 features") / "education-13-00017-v3.pdf",
        "title": "Predicting Student Performance Using Clickstream Data and Machine Learning",
        "reference_hint": "Liu et al. (2022)",
        "venue": "Education Sciences",
        "year": 2022,
        "selected_claim": "Activity-specific clickstream features carry strong predictive signal for student performance in OULAD-style settings.",
    },
    {
        "section": "Demographic and Enrollment Features",
        "report_section": "Feature engineering",
        "pdf_path": Path("Demographic & Enrollment Features")
        / "PIIS2405844024069913.pdf",
        "title": "Predictive modelling of student dropout risk: Practical insights from a South Korean distance university",
        "reference_hint": "Seo et al. (2024)",
        "venue": "Heliyon",
        "year": 2024,
        "selected_claim": "Demographic and enrollment information improves early student risk prediction when combined with activity data.",
    },
    {
        "section": "Early Assessment Features",
        "report_section": "Feature engineering",
        "pdf_path": Path("Early Assessment Features")
        / "Early_Prediction_of_Dropout_and_Final_Exam_Performance_in_an_Online_Statistics_Course.pdf",
        "title": "Early Prediction of Dropout and Final Exam Performance in an Online Statistics Course",
        "reference_hint": "Figueroa-Cañas et al. (2020)",
        "venue": "IEEE Revista Iberoamericana de Tecnologias del Aprendizaje",
        "year": 2020,
        "selected_claim": "Early formative assessment behavior can improve dropout and performance prediction under a strict early-course observation window.",
    },
    {
        "section": "Logistic Regression Baseline",
        "report_section": "Model 1 baseline",
        "pdf_path": Path("Logistic Regression as Taught Baseline")
        / "Predicting student dropout  A machine learning approach.pdf",
        "title": "Predicting student dropout: A machine learning approach",
        "reference_hint": "Kemper et al. (2020)",
        "venue": "European Journal of Higher Education",
        "year": 2020,
        "selected_claim": "Logistic regression is a credible, interpretable baseline for dropout prediction on institutional tabular data.",
    },
    {
        "section": "CatBoost in Education",
        "report_section": "Model 2 self-learned method",
        "pdf_path": Path("CatBoost as Self-Learned Model (Education-Specific)")
        / "s41598-025-93918-1.pdf",
        "title": "Student dropout prediction through machine learning optimization: insights from moodle log data",
        "reference_hint": "Marcolino et al. (2025)",
        "venue": "Scientific Reports",
        "year": 2025,
        "selected_claim": "CatBoost performs strongly on higher-education LMS dropout prediction tasks using behavioral signals.",
    },
    {
        "section": "CatBoost Technical Justification",
        "report_section": "Model 2 self-learned method",
        "pdf_path": Path("CatBoost Technical Justification (FallbackAlgorithmic)")
        / "s40537-020-00369-8.pdf",
        "title": "CatBoost for big data: an interdisciplinary review",
        "reference_hint": "Hancock and Khoshgoftaar (2020)",
        "venue": "Journal of Big Data",
        "year": 2020,
        "selected_claim": "CatBoost is especially suitable for heterogeneous tabular data with categorical variables and missingness.",
    },
    {
        "section": "Class Weighting and Imbalanced Learning",
        "report_section": "Model 2 self-learned method",
        "pdf_path": Path("Class Weighting & Imbalanced Learning")
        / "applsci-09-03093.pdf",
        "title": "The Machine Learning-Based Dropout Early Warning System for Improving the Performance of Dropout Prediction",
        "reference_hint": "Lee et al. (2019)",
        "venue": "Applied Sciences",
        "year": 2019,
        "selected_claim": "Imbalance-aware training should be evaluated with recall-sensitive metrics in student early warning systems.",
    },
    {
        "section": "Probability Calibration",
        "report_section": "Evaluation and thresholding",
        "pdf_path": Path("Probability Calibration") / "1706.04599v2.pdf",
        "title": "On Calibration of Modern Neural Networks",
        "reference_hint": "Guo et al. (2017)",
        "venue": "Proceedings of ICML",
        "year": 2017,
        "selected_claim": "Raw classifier probabilities are often miscalibrated, and sigmoid-style post-hoc calibration can improve decision-quality probabilities.",
    },
    {
        "section": "Threshold Choice",
        "report_section": "Evaluation and thresholding",
        "pdf_path": Path("Threshold Choice for Intervention")
        / "Receiver Operating Characteristic  ROC  Area Under the Curve  AUC   A Diagnostic Measure for Evaluating the Accuracy of Predictors of Education Outcom.pdf",
        "title": "Receiver Operating Characteristic (ROC) Area Under the Curve (AUC): A Diagnostic Measure for Evaluating the Accuracy of Predictors of Education Outcomes",
        "reference_hint": "Bowers and Zhou (2019)",
        "venue": "Journal of Educational Students Placed at Risk",
        "year": 2019,
        "selected_claim": "Education early warning systems should choose decision thresholds using sensitivity and specificity trade-offs rather than assuming 0.5 is optimal.",
    },
    {
        "section": "Evaluation Metrics",
        "report_section": "Evaluation and thresholding",
        "pdf_path": Path("Evaluation Metrics") / "s13040-023-00322-4.pdf",
        "title": "The Matthews correlation coefficient (MCC) should replace the ROC AUC as the standard metric",
        "reference_hint": "Chicco and Jurman (2023)",
        "venue": "BioData Mining",
        "year": 2023,
        "selected_claim": "Metric choice changes interpretation, so probability-aware and threshold-aware evaluation should accompany accuracy-like summaries.",
    },
    {
        "section": "Leakage-Safe Temporal Cutoff",
        "report_section": "Leakage rules",
        "pdf_path": Path("Leakage-Safe Temporal Cutoff") / "2511.11866v1.pdf",
        "title": "A Leakage-Aware Data Layer For Student Analytics: The Capire Framework",
        "reference_hint": "Paz (2025)",
        "venue": "Proceedings of LAK",
        "year": 2025,
        "selected_claim": "Observation windows and prediction horizons should be separated explicitly to avoid temporal leakage in student analytics.",
    },
    {
        "section": "Ethics, Fairness, and Privacy",
        "report_section": "Ethics",
        "pdf_path": Path("Ethics, Fairness & Privacy") / "Ethical and TrustWorthy LA.pdf",
        "title": "How do the existing fairness metrics and unfairness mitigation algorithms contribute to ethical learning analytics?",
        "reference_hint": "Deho et al. (2021)",
        "venue": "British Journal of Educational Technology",
        "year": 2021,
        "selected_claim": "Fairness and bias should be considered whenever demographic attributes are used in learning analytics models.",
    },
    {
        "section": "Interpretability vs Performance",
        "report_section": "Discussion",
        "pdf_path": Path("Interpretability vs. Performance") / "s40593-023-00331-8.pdf",
        "title": "Interpretable Dropout Prediction: Towards XAI-Based Personalized Intervention",
        "reference_hint": "Nagy et al. (2023)",
        "venue": "International Journal of Artificial Intelligence in Education",
        "year": 2023,
        "selected_claim": "High-performing ensemble models benefit from explanation methods when used for student intervention workflows.",
    },
]


@dataclass
class PipelineConfig:
    project_root: Path
    artifacts_dir: Path
    plots_dir: Path
    submission_dir: Path
    submission_plots_dir: Path

    @classmethod
    def from_project_root(cls, project_root: Path | str) -> "PipelineConfig":
        root = Path(project_root).resolve()
        artifacts_dir = root / "artifacts"
        plots_dir = artifacts_dir / "plots"
        submission_dir = artifacts_dir / "submission"
        submission_plots_dir = submission_dir / "plots"
        return cls(
            project_root=root,
            artifacts_dir=artifacts_dir,
            plots_dir=plots_dir,
            submission_dir=submission_dir,
            submission_plots_dir=submission_plots_dir,
        )


def ensure_directories(config: PipelineConfig) -> None:
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    config.plots_dir.mkdir(parents=True, exist_ok=True)
    config.submission_dir.mkdir(parents=True, exist_ok=True)
    config.submission_plots_dir.mkdir(parents=True, exist_ok=True)


def load_student_info(config: PipelineConfig) -> pd.DataFrame:
    path = config.project_root / "studentInfo.csv"
    df = pd.read_csv(
        path,
        dtype={
            "code_module": "string",
            "code_presentation": "string",
            "id_student": "Int64",
            "gender": "string",
            "region": "string",
            "highest_education": "string",
            "imd_band": "string",
            "age_band": "string",
            "num_of_prev_attempts": "Int64",
            "studied_credits": "Int64",
            "disability": "string",
            "final_result": "string",
        },
    )
    df["binary_target"] = np.where(
        df["final_result"].isin(["Pass", "Distinction"]), "Favourable", "Unfavourable"
    )
    df["target"] = np.where(df["binary_target"].eq("Favourable"), 1, 0).astype(int)
    return df


def load_registration_features(config: PipelineConfig) -> pd.DataFrame:
    path = config.project_root / "studentRegistration.csv"
    df = pd.read_csv(
        path,
        dtype={
            "code_module": "string",
            "code_presentation": "string",
            "id_student": "Int64",
        },
    )
    df["date_registration"] = pd.to_numeric(df["date_registration"], errors="coerce")
    df["registered_before_start_days"] = (
        -df["date_registration"].clip(upper=0).fillna(0)
    )
    df["late_registration_flag"] = (
        df["date_registration"].fillna(-9999).gt(0).astype(int)
    )
    keep_columns = [
        "code_module",
        "code_presentation",
        "id_student",
        "date_registration",
        "registered_before_start_days",
        "late_registration_flag",
    ]
    return df[keep_columns]


def build_vle_features(config: PipelineConfig) -> tuple[pd.DataFrame, dict[str, int]]:
    vle_path = config.project_root / "vle.csv"
    student_vle_path = config.project_root / "studentVle.csv"

    vle_meta = pd.read_csv(
        vle_path,
        dtype={
            "id_site": "Int64",
            "code_module": "string",
            "code_presentation": "string",
            "activity_type": "string",
        },
        usecols=[
            "id_site",
            "code_module",
            "code_presentation",
            "activity_type",
            "week_from",
            "week_to",
        ],
    )
    if "recommended" in vle_meta.columns:
        vle_meta["recommended_flag"] = (
            vle_meta["recommended"].fillna("").astype(str).str.strip().ne("")
        )
        vle_meta = vle_meta.drop(columns=["recommended"])
    else:
        vle_meta["recommended_flag"] = False

    group_cols = ["code_module", "code_presentation", "id_student"]
    week4_frames: list[pd.DataFrame] = []
    raw_student_vle_rows = 0
    week4_student_vle_rows = 0

    chunks = pd.read_csv(
        student_vle_path,
        dtype={
            "code_module": "string",
            "code_presentation": "string",
            "id_student": "Int64",
            "id_site": "Int64",
            "date": "Int64",
            "sum_click": "Int64",
        },
        usecols=[
            "code_module",
            "code_presentation",
            "id_student",
            "id_site",
            "date",
            "sum_click",
        ],
        chunksize=1_000_000,
    )

    for chunk in chunks:
        raw_student_vle_rows += len(chunk)
        chunk = chunk.loc[
            chunk["date"].between(WEEK4_START_DAY, WEEK4_END_DAY, inclusive="both")
        ].copy()
        if chunk.empty:
            continue
        week4_student_vle_rows += len(chunk)
        week4_frames.append(chunk)

    if not week4_frames:
        raise RuntimeError("No week-4 VLE events were found in studentVle.csv.")

    week4_events = pd.concat(week4_frames, ignore_index=True)
    exact_event_key = group_cols + ["id_site", "date", "sum_click"]
    week4_exact_duplicates_removed = int(
        week4_events.duplicated(subset=exact_event_key).sum()
    )
    week4_events = week4_events.drop_duplicates(subset=exact_event_key).reset_index(drop=True)
    week4_events["sum_click"] = pd.to_numeric(
        week4_events["sum_click"], errors="coerce"
    ).fillna(0)
    week4_events = week4_events.merge(
        vle_meta,
        on=["code_module", "code_presentation", "id_site"],
        how="left",
    )
    week4_events["activity_type"] = (
        week4_events["activity_type"].fillna("other_activity").str.lower().str.strip()
    )
    week4_events["activity_group"] = np.where(
        week4_events["activity_type"].isin(EXPECTED_ACTIVITY_TYPES),
        week4_events["activity_type"],
        "other_activity",
    )

    vle_features = (
        week4_events.groupby(group_cols, as_index=False)
        .agg(
            total_clicks_w4=("sum_click", "sum"),
            first_activity_day_w4=("date", "min"),
            last_activity_day_w4=("date", "max"),
        )
        .reset_index(drop=True)
    )

    active_days = (
        week4_events[group_cols + ["date"]]
        .drop_duplicates()
        .groupby(group_cols, as_index=False)
        .size()
        .rename(columns={"size": "active_days_w4"})
    )
    unique_sites = (
        week4_events[group_cols + ["id_site"]]
        .drop_duplicates()
        .groupby(group_cols, as_index=False)
        .size()
        .rename(columns={"size": "unique_sites_w4"})
    )
    vle_features = vle_features.merge(active_days, on=group_cols, how="left")
    vle_features = vle_features.merge(unique_sites, on=group_cols, how="left")

    recommended_features = (
        week4_events.loc[week4_events["recommended_flag"].fillna(False)]
        .groupby(group_cols, as_index=False)["sum_click"]
        .sum()
        .rename(columns={"sum_click": "recommended_clicks_w4"})
    )
    if recommended_features.empty:
        vle_features["recommended_clicks_w4"] = 0
    else:
        vle_features = vle_features.merge(recommended_features, on=group_cols, how="left")

    activity_features = (
        week4_events.pivot_table(
            index=group_cols,
            columns="activity_group",
            values="sum_click",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )
    activity_features.columns = [
        str(column) if isinstance(column, str) else column
        for column in activity_features.columns
    ]
    rename_map = {
        activity_name: f"{activity_name}_clicks_w4"
        for activity_name in EXPECTED_ACTIVITY_TYPES + ["other_activity"]
        if activity_name in activity_features.columns
    }
    activity_features = activity_features.rename(columns=rename_map)
    if not activity_features.empty:
        vle_features = vle_features.merge(activity_features, on=group_cols, how="left")

    for activity_name in EXPECTED_ACTIVITY_TYPES + ["other_activity"]:
        column = f"{activity_name}_clicks_w4"
        if column not in vle_features.columns:
            vle_features[column] = 0

    vle_features["recommended_clicks_w4"] = vle_features["recommended_clicks_w4"].fillna(0)
    vle_features["clicks_per_active_day_w4"] = np.where(
        vle_features["active_days_w4"].gt(0),
        vle_features["total_clicks_w4"] / vle_features["active_days_w4"],
        0.0,
    )
    vle_features["days_since_last_activity_w4"] = np.where(
        vle_features["last_activity_day_w4"].notna(),
        WEEK4_END_DAY - vle_features["last_activity_day_w4"],
        np.nan,
    )
    vle_features["recommended_click_ratio_w4"] = np.where(
        vle_features["total_clicks_w4"].gt(0),
        vle_features["recommended_clicks_w4"] / vle_features["total_clicks_w4"],
        0.0,
    )
    counts = {
        "vle": int(len(vle_meta)),
        "studentVle": int(raw_student_vle_rows),
        "studentVle_week4_events": int(week4_student_vle_rows),
        "studentVle_week4_exact_duplicates_removed": int(week4_exact_duplicates_removed),
    }
    return vle_features.drop(columns=["last_activity_day_w4"]), counts


def build_assessment_features(
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    assessments_path = config.project_root / "assessments.csv"
    student_assessment_path = config.project_root / "studentAssessment.csv"
    group_cols = ["code_module", "code_presentation", "id_student"]

    assessments = pd.read_csv(
        assessments_path,
        dtype={
            "code_module": "string",
            "code_presentation": "string",
            "id_assessment": "Int64",
            "assessment_type": "string",
        },
        usecols=[
            "code_module",
            "code_presentation",
            "id_assessment",
            "assessment_type",
            "date",
            "weight",
        ],
    )
    assessments["date"] = pd.to_numeric(assessments["date"], errors="coerce")
    assessments["weight"] = pd.to_numeric(assessments["weight"], errors="coerce")
    early_assessments = assessments.loc[
        assessments["date"].between(WEEK4_START_DAY, WEEK4_END_DAY, inclusive="both")
    ].copy()

    due_features = (
        early_assessments.groupby(["code_module", "code_presentation"], as_index=False)
        .agg(
            early_assessment_due_count=("id_assessment", "nunique"),
            weighted_due_by_w4=("weight", "sum"),
        )
        .reset_index(drop=True)
    )

    submissions = pd.read_csv(
        student_assessment_path,
        dtype={"id_assessment": "Int64", "id_student": "Int64", "is_banked": "Int64"},
        usecols=["id_assessment", "id_student", "date_submitted", "is_banked", "score"],
    )
    submissions["date_submitted"] = pd.to_numeric(
        submissions["date_submitted"], errors="coerce"
    )
    counts = {
        "assessments": int(len(assessments)),
        "assessments_due_by_week4": int(len(early_assessments)),
        "studentAssessment": int(len(submissions)),
    }

    merged = early_assessments.merge(submissions, on="id_assessment", how="left")
    merged = merged.loc[
        merged["date_submitted"].between(WEEK4_START_DAY, WEEK4_END_DAY, inclusive="both")
    ].copy()
    if merged.empty:
        keep_columns = group_cols + [
            "early_assessment_submitted_count",
            "submitted_any_early_assessment",
            "on_time_submission_count",
            "late_submission_count",
            "mean_submission_delay_days",
            "weighted_submitted_by_w4",
        ]
        counts["studentAssessment_submitted_by_week4"] = 0
        return due_features, pd.DataFrame(columns=keep_columns), counts

    merged["submission_delay_days"] = merged["date_submitted"] - merged["date"]
    merged["on_time_flag"] = merged["submission_delay_days"].le(0).astype(int)
    merged["late_flag"] = merged["submission_delay_days"].gt(0).astype(int)

    submitted_features = (
        merged.groupby(group_cols, as_index=False)
        .agg(
            early_assessment_submitted_count=("id_assessment", "nunique"),
            on_time_submission_count=("on_time_flag", "sum"),
            late_submission_count=("late_flag", "sum"),
            mean_submission_delay_days=("submission_delay_days", "mean"),
            weighted_submitted_by_w4=("weight", "sum"),
        )
        .reset_index(drop=True)
    )
    submitted_features["submitted_any_early_assessment"] = (
        submitted_features["early_assessment_submitted_count"].gt(0).astype(int)
    )
    counts["studentAssessment_submitted_by_week4"] = int(len(merged))

    return due_features, submitted_features, counts


def build_feature_table(config: PipelineConfig) -> tuple[pd.DataFrame, dict[str, int]]:
    student_info = load_student_info(config)
    row_counts = {
        "studentInfo": int(len(student_info)),
    }
    master_columns = [
        "code_module",
        "code_presentation",
        "id_student",
        "gender",
        "region",
        "highest_education",
        "imd_band",
        "age_band",
        "num_of_prev_attempts",
        "studied_credits",
        "disability",
        "final_result",
        "binary_target",
        "target",
    ]
    feature_table = student_info[master_columns].copy()

    registration = load_registration_features(config)
    row_counts["studentRegistration"] = int(len(registration))
    feature_table = feature_table.merge(
        registration,
        on=["code_module", "code_presentation", "id_student"],
        how="left",
    )

    vle_features, vle_counts = build_vle_features(config)
    row_counts.update(vle_counts)
    row_counts["studentVle_filtered_feature_rows"] = int(len(vle_features))
    feature_table = feature_table.merge(
        vle_features,
        on=["code_module", "code_presentation", "id_student"],
        how="left",
    )

    due_features, submitted_features, assessment_counts = build_assessment_features(config)
    row_counts.update(assessment_counts)
    row_counts["assessments_due_rows"] = int(len(due_features))
    row_counts["studentAssessment_submitted_feature_rows"] = int(len(submitted_features))

    feature_table = feature_table.merge(
        due_features,
        on=["code_module", "code_presentation"],
        how="left",
    )
    feature_table = feature_table.merge(
        submitted_features,
        on=["code_module", "code_presentation", "id_student"],
        how="left",
    )

    zero_fill_columns = [
        "total_clicks_w4",
        "active_days_w4",
        "unique_sites_w4",
        "clicks_per_active_day_w4",
        "recommended_clicks_w4",
        "recommended_click_ratio_w4",
        "forumng_clicks_w4",
        "homepage_clicks_w4",
        "oucontent_clicks_w4",
        "quiz_clicks_w4",
        "resource_clicks_w4",
        "subpage_clicks_w4",
        "url_clicks_w4",
        "page_clicks_w4",
        "other_activity_clicks_w4",
        "early_assessment_due_count",
        "weighted_due_by_w4",
        "early_assessment_submitted_count",
        "submitted_any_early_assessment",
        "on_time_submission_count",
        "late_submission_count",
        "weighted_submitted_by_w4",
    ]
    for column in zero_fill_columns:
        if column in feature_table.columns:
            feature_table[column] = feature_table[column].fillna(0)

    feature_table["registered_before_start_days"] = feature_table[
        "registered_before_start_days"
    ].fillna(0)
    feature_table["late_registration_flag"] = feature_table["late_registration_flag"].fillna(0)

    if "mean_submission_delay_days" in feature_table.columns:
        feature_table["mean_submission_delay_days"] = feature_table[
            "mean_submission_delay_days"
        ].fillna(0)

    feature_table = feature_table.sort_values(
        ["code_module", "code_presentation", "id_student"]
    ).reset_index(drop=True)
    validate_feature_table(feature_table)
    return feature_table, row_counts


def validate_feature_table(feature_table: pd.DataFrame) -> None:
    key_cols = ["code_module", "code_presentation", "id_student"]
    if feature_table.duplicated(key_cols).any():
        raise ValueError("Feature table contains duplicate student-module-presentation rows.")
    if len(feature_table) == 0:
        raise ValueError("Feature table is empty.")


def numeric_signal_summary(feature_table: pd.DataFrame) -> pd.DataFrame:
    candidate_columns = [
        "total_clicks_w4",
        "active_days_w4",
        "clicks_per_active_day_w4",
        "recommended_click_ratio_w4",
        "registered_before_start_days",
        "early_assessment_submitted_count",
        "weighted_submitted_by_w4",
        "days_since_last_activity_w4",
    ]
    rows = []
    for column in candidate_columns:
        if column not in feature_table.columns:
            continue
        favourable_mean = feature_table.loc[
            feature_table["target"].eq(1), column
        ].mean(skipna=True)
        unfavourable_mean = feature_table.loc[
            feature_table["target"].eq(0), column
        ].mean(skipna=True)
        rows.append(
            {
                "feature": column,
                "mean_favourable": favourable_mean,
                "mean_unfavourable": unfavourable_mean,
                "absolute_mean_gap": abs(favourable_mean - unfavourable_mean),
            }
        )
    signal = pd.DataFrame(rows).sort_values(
        "absolute_mean_gap", ascending=False
    ).reset_index(drop=True)
    return signal


def save_feature_dictionary(config: PipelineConfig, feature_table: pd.DataFrame) -> None:
    descriptions = {
        "code_module": "OULAD module code.",
        "code_presentation": "OULAD presentation code.",
        "id_student": "Student identifier, retained for joins but excluded from modeling.",
        "binary_target": "Binary label: Favourable = Pass/Distinction, Unfavourable = Fail/Withdrawn.",
        "target": "Numeric target encoded as 1 for Favourable and 0 for Unfavourable.",
        "date_registration": "Registration day relative to presentation start.",
        "registered_before_start_days": "Days registered before presentation start, floored at zero.",
        "late_registration_flag": "1 if the student registered after day 0, else 0.",
        "total_clicks_w4": "Total VLE clicks from day 0 to day 27.",
        "active_days_w4": "Distinct active click days from day 0 to day 27.",
        "unique_sites_w4": "Distinct VLE sites visited from day 0 to day 27.",
        "clicks_per_active_day_w4": "Average clicks per active day in week 4 window.",
        "first_activity_day_w4": "First observed VLE activity day in the week 4 window.",
        "days_since_last_activity_w4": "Recency feature: 27 minus the last active day in the week 4 window.",
        "recommended_clicks_w4": "Clicks on VLE resources flagged as recommended.",
        "recommended_click_ratio_w4": "Recommended clicks divided by total clicks.",
        "early_assessment_due_count": "Number of assessments due by day 27 for the student's module presentation.",
        "early_assessment_submitted_count": "Number of those early assessments submitted by day 27.",
        "submitted_any_early_assessment": "Binary flag for at least one early assessment submission.",
        "on_time_submission_count": "Count of early assessments submitted on or before their due date.",
        "late_submission_count": "Count of early assessments submitted after their due date but by day 27.",
        "mean_submission_delay_days": "Average delay between submission day and due day for early assessments.",
        "weighted_due_by_w4": "Total percentage weight of assessments due by day 27.",
        "weighted_submitted_by_w4": "Total percentage weight of early assessments submitted by day 27.",
    }
    lines = [
        "# Week-4 Feature Dictionary",
        "",
        "This artifact documents the reusable week-4 feature table generated from the OULAD source files.",
        "",
        "| Column | Description |",
        "| --- | --- |",
    ]
    for column in feature_table.columns:
        description = descriptions.get(column, "See notebook for usage context.")
        lines.append(f"| {column} | {description} |")
    (config.artifacts_dir / "feature_dictionary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def extract_pdf_details(pdf_path: Path) -> dict[str, Any]:
    details: dict[str, Any] = {
        "metadata_title": "",
        "metadata_author": "",
        "metadata_subject": "",
        "doi": "",
        "first_page_excerpt": "",
    }
    try:
        reader = PdfReader(str(pdf_path))
        metadata = reader.metadata or {}
        details["metadata_title"] = str(metadata.get("/Title", "") or "")
        details["metadata_author"] = str(metadata.get("/Author", "") or "")
        details["metadata_subject"] = str(metadata.get("/Subject", "") or "")
        details["doi"] = str(metadata.get("/doi", "") or "")
        excerpt = "\n".join((page.extract_text() or "") for page in reader.pages[:2])
        excerpt = re.sub(r"\s+", " ", excerpt).strip()
        details["first_page_excerpt"] = excerpt[:500]
        if not details["doi"]:
            match = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", excerpt, re.IGNORECASE)
            if match:
                details["doi"] = match.group(0)
    except Exception as exc:  # pragma: no cover
        details["first_page_excerpt"] = f"Metadata extraction failed: {exc}"
    return details


def save_literature_matrix(config: PipelineConfig) -> pd.DataFrame:
    rows = []
    for paper in PAPER_CATALOG:
        pdf_path = config.project_root / paper["pdf_path"]
        extracted = extract_pdf_details(pdf_path)
        title_match = paper["title"].lower() in extracted["first_page_excerpt"].lower()
        rows.append(
            {
                "section": paper["section"],
                "report_section": paper["report_section"],
                "pdf_path": str(pdf_path.relative_to(config.project_root)),
                "title": paper["title"],
                "reference_hint": paper["reference_hint"],
                "venue": paper["venue"],
                "year": paper["year"],
                "doi": extracted["doi"],
                "selected_claim": paper["selected_claim"],
                "metadata_title": extracted["metadata_title"],
                "metadata_author": extracted["metadata_author"],
                "title_found_in_excerpt": bool(title_match),
                "evidence_note": "Verify final citation details against the PDF title page before submission.",
                "first_page_excerpt": extracted["first_page_excerpt"],
            }
        )
    literature = pd.DataFrame(rows)
    literature.to_csv(config.artifacts_dir / "literature_matrix.csv", index=False)

    markdown_lines = [
        "# Literature Matrix",
        "",
        "This matrix links each selected paper to its job in the report and records DOI or PDF metadata where available.",
        "",
        "| Section | Report Section | Reference Hint | Year | DOI | Claim for Report | PDF |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in literature.itertuples(index=False):
        doi_value = row.doi if row.doi else "Verify in PDF"
        markdown_lines.append(
            "| {section} | {report_section} | {reference_hint} | {year} | {doi} | {selected_claim} | {pdf_path} |".format(
                section=row.section,
                report_section=row.report_section,
                reference_hint=row.reference_hint,
                year=row.year,
                doi=doi_value,
                selected_claim=row.selected_claim,
                pdf_path=row.pdf_path,
            )
        )
    (config.artifacts_dir / "literature_matrix.md").write_text(
        "\n".join(markdown_lines) + "\n",
        encoding="utf-8",
    )
    return literature


def save_eda_outputs(
    config: PipelineConfig,
    feature_table: pd.DataFrame,
    row_counts: dict[str, int],
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    row_count_df = pd.DataFrame(
        [{"table": table_name, "row_count": row_count} for table_name, row_count in row_counts.items()]
    )
    row_count_path = config.artifacts_dir / "row_counts.csv"
    row_count_df.to_csv(row_count_path, index=False)
    outputs["row_counts"] = row_count_path

    missingness = (
        feature_table.isna().sum().sort_values(ascending=False).rename("missing_count").reset_index()
    )
    missingness = missingness.rename(columns={"index": "column"})
    missingness_path = config.artifacts_dir / "missing_value_counts.csv"
    missingness.to_csv(missingness_path, index=False)
    outputs["missing_value_counts"] = missingness_path

    class_balance = (
        feature_table.groupby("binary_target", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("binary_target")
        .reset_index(drop=True)
    )
    class_balance_path = config.artifacts_dir / "class_balance.csv"
    class_balance.to_csv(class_balance_path, index=False)
    outputs["class_balance"] = class_balance_path

    signal = numeric_signal_summary(feature_table)
    signal_path = config.artifacts_dir / "feature_signal_summary.csv"
    signal.to_csv(signal_path, index=False)
    outputs["feature_signal_summary"] = signal_path

    sns.set_theme(style="whitegrid")

    hist_path = config.plots_dir / "total_clicks_w4_by_target.png"
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=feature_table,
        x="total_clicks_w4",
        hue="binary_target",
        bins=40,
        stat="density",
        common_norm=False,
        element="step",
    )
    plt.title("Week-4 Total Clicks by Binary Outcome")
    plt.xlabel("total_clicks_w4")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(hist_path, dpi=200)
    plt.close()
    outputs["total_clicks_histogram"] = hist_path

    activity_columns = [f"{activity}_clicks_w4" for activity in EXPECTED_ACTIVITY_TYPES + ["other_activity"]]
    activity_means = feature_table[activity_columns].mean().sort_values(ascending=False).reset_index()
    activity_means.columns = ["activity_feature", "mean_clicks"]
    activity_bar_path = config.plots_dir / "activity_type_mean_clicks.png"
    plt.figure(figsize=(11, 6))
    sns.barplot(data=activity_means, x="mean_clicks", y="activity_feature", hue="activity_feature", palette="crest", legend=False)
    plt.title("Mean Week-4 Clicks by Activity Feature")
    plt.xlabel("Mean clicks")
    plt.ylabel("Activity feature")
    plt.tight_layout()
    plt.savefig(activity_bar_path, dpi=200)
    plt.close()
    outputs["activity_type_bar_chart"] = activity_bar_path

    return outputs


def get_model_columns(feature_table: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    exclude_columns = {
        "id_student",
        "final_result",
        "binary_target",
        "target",
    }
    categorical_columns = CATEGORICAL_MODEL_COLUMNS.copy()
    feature_columns = [column for column in feature_table.columns if column not in exclude_columns]
    numeric_columns = [column for column in feature_columns if column not in categorical_columns]
    return feature_columns, categorical_columns, numeric_columns


def normalize_model_frame(
    X: pd.DataFrame, categorical_columns: list[str], numeric_columns: list[str]
) -> pd.DataFrame:
    frame = X.copy()
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
    for column in categorical_columns:
        frame[column] = frame[column].astype(object)
        frame.loc[pd.isna(frame[column]), column] = np.nan
    return frame


def split_dataset(
    feature_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    feature_columns, _, _ = get_model_columns(feature_table)
    X = feature_table[feature_columns].copy()
    y = feature_table["target"].copy()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def build_logistic_pipeline(
    categorical_columns: list[str], numeric_columns: list[str]
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )


def prepare_catboost_inputs(
    X: pd.DataFrame, categorical_columns: list[str]
) -> pd.DataFrame:
    catboost_frame = X.copy()
    for column in categorical_columns:
        catboost_frame[column] = catboost_frame[column].fillna("Missing").astype(str)
    return catboost_frame


def calculate_threshold_metrics(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    model_name: str,
    split_name: str,
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "model": model_name,
        "split": split_name,
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, predictions),
        "log_loss": log_loss(y_true, probabilities, labels=[0, 1]),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "specificity": specificity,
        "fpr": fpr,
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def choose_best_threshold(sweep: pd.DataFrame) -> float:
    ordered = sweep.sort_values(
        by=["f1", "recall", "fpr", "threshold"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    return float(ordered.loc[0, "threshold"])


def save_roc_plot(
    config: PipelineConfig,
    y_true: pd.Series,
    probability_map: dict[str, np.ndarray],
) -> Path:
    path = config.plots_dir / "roc_curves_test.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, probabilities in probability_map.items():
        fpr_values, tpr_values, _ = roc_curve(y_true, probabilities)
        auc_value = roc_auc_score(y_true, probabilities)
        ax.plot(fpr_values, tpr_values, label=f"{label} (AUC={auc_value:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves on the Test Split")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.margins(x=0, y=0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_threshold_plot(config: PipelineConfig, threshold_sweep: pd.DataFrame) -> Path:
    path = config.plots_dir / "threshold_sweep_f1_precision_recall.png"
    plot_data = threshold_sweep.loc[
        threshold_sweep["model"].isin(["Logistic Regression", "Calibrated CatBoost"])
        & threshold_sweep["split"].eq("validation")
    ].copy()
    melted = plot_data.melt(
        id_vars=["model", "threshold"],
        value_vars=["precision", "recall", "f1"],
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=melted,
        x="threshold",
        y="value",
        hue="metric",
        style="model",
        markers=True,
        dashes=False,
        ax=ax,
    )
    ax.set_title("Validation Threshold Sweep")
    ax.set_ylabel("Score")
    ax.set_xlabel("Threshold")
    min_threshold = float(plot_data["threshold"].min())
    max_threshold = float(plot_data["threshold"].max())
    ax.set_xlim(min_threshold - 0.01, max_threshold + 0.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xticks(sorted(plot_data["threshold"].unique()))
    ax.margins(x=0, y=0)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def fit_and_evaluate_models(
    config: PipelineConfig, feature_table: pd.DataFrame
) -> dict[str, Any]:
    feature_columns, categorical_columns, numeric_columns = get_model_columns(feature_table)
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_dataset(feature_table)
    X_train = normalize_model_frame(X_train, categorical_columns, numeric_columns)
    X_valid = normalize_model_frame(X_valid, categorical_columns, numeric_columns)
    X_test = normalize_model_frame(X_test, categorical_columns, numeric_columns)

    logistic = build_logistic_pipeline(categorical_columns, numeric_columns)
    logistic.fit(X_train, y_train)
    logistic_valid_proba = logistic.predict_proba(X_valid)[:, 1]
    logistic_test_proba = logistic.predict_proba(X_test)[:, 1]

    cat_train_core, cat_calib, y_cat_train_core, y_cat_calib = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )
    cat_train_core_prepared = prepare_catboost_inputs(cat_train_core, categorical_columns)
    cat_calib_prepared = prepare_catboost_inputs(cat_calib, categorical_columns)
    X_valid_prepared = prepare_catboost_inputs(X_valid, categorical_columns)
    X_test_prepared = prepare_catboost_inputs(X_test, categorical_columns)

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
        cat_train_core_prepared,
        y_cat_train_core,
        cat_features=categorical_columns,
        eval_set=(cat_calib_prepared, y_cat_calib),
        use_best_model=True,
        verbose=False,
    )
    catboost_valid_proba = catboost.predict_proba(X_valid_prepared)[:, 1]
    catboost_test_proba = catboost.predict_proba(X_test_prepared)[:, 1]

    calibrated_catboost = CalibratedClassifierCV(
        estimator=FrozenEstimator(catboost), method="sigmoid", cv=None
    )
    calibrated_catboost.fit(cat_calib_prepared, y_cat_calib)
    calibrated_valid_proba = calibrated_catboost.predict_proba(X_valid_prepared)[:, 1]
    calibrated_test_proba = calibrated_catboost.predict_proba(X_test_prepared)[:, 1]

    thresholds = [round(value, 1) for value in np.arange(0.1, 1.0, 0.1)]
    sweep_rows = []
    for threshold in thresholds:
        sweep_rows.append(
            calculate_threshold_metrics(
                y_valid, logistic_valid_proba, threshold, "Logistic Regression", "validation"
            )
        )
        sweep_rows.append(
            calculate_threshold_metrics(
                y_valid,
                calibrated_valid_proba,
                threshold,
                "Calibrated CatBoost",
                "validation",
            )
        )
    threshold_sweep = pd.DataFrame(sweep_rows)

    logistic_best_threshold = choose_best_threshold(
        threshold_sweep.loc[threshold_sweep["model"].eq("Logistic Regression")]
    )
    calibrated_best_threshold = choose_best_threshold(
        threshold_sweep.loc[threshold_sweep["model"].eq("Calibrated CatBoost")]
    )

    comparison_rows = [
        calculate_threshold_metrics(
            y_test, logistic_test_proba, 0.5, "Logistic Regression", "test"
        ),
        calculate_threshold_metrics(
            y_test,
            logistic_test_proba,
            logistic_best_threshold,
            "Logistic Regression (tuned threshold)",
            "test",
        ),
        calculate_threshold_metrics(
            y_test, catboost_test_proba, 0.5, "CatBoost", "test"
        ),
        calculate_threshold_metrics(
            y_test,
            calibrated_test_proba,
            calibrated_best_threshold,
            "Calibrated CatBoost (tuned threshold)",
            "test",
        ),
    ]
    comparison = pd.DataFrame(comparison_rows)

    threshold_sweep_path = config.artifacts_dir / "threshold_sweep.csv"
    threshold_sweep.to_csv(threshold_sweep_path, index=False)

    comparison_path = config.artifacts_dir / "metrics_summary.csv"
    comparison.to_csv(comparison_path, index=False)

    roc_plot = save_roc_plot(
        config,
        y_test,
        {
            "Logistic Regression": logistic_test_proba,
            "CatBoost": catboost_test_proba,
            "Calibrated CatBoost": calibrated_test_proba,
        },
    )
    threshold_plot = save_threshold_plot(config, threshold_sweep)

    return {
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "comparison_table": comparison,
        "threshold_sweep": threshold_sweep,
        "threshold_sweep_path": threshold_sweep_path,
        "metrics_summary_path": comparison_path,
        "roc_plot_path": roc_plot,
        "threshold_plot_path": threshold_plot,
        "logistic_best_threshold": logistic_best_threshold,
        "calibrated_best_threshold": calibrated_best_threshold,
        "logistic_validation_log_loss": log_loss(y_valid, logistic_valid_proba, labels=[0, 1]),
        "raw_catboost_validation_log_loss": log_loss(
            y_valid, catboost_valid_proba, labels=[0, 1]
        ),
        "calibrated_catboost_validation_log_loss": log_loss(
            y_valid, calibrated_valid_proba, labels=[0, 1]
        ),
        "test_size": len(X_test),
        "validation_size": len(X_valid),
        "train_size": len(X_train),
    }


def save_integration_note(config: PipelineConfig) -> Path:
    path = config.artifacts_dir / "integration_note.md"
    content = "\n".join(
        [
            "# Reusable Artifact Integration Note",
            "",
            "- Artifact: `week4_feature_table.parquet`",
            "- Grain: one row per `(code_module, code_presentation, id_student)`",
            "- Target mapping: Pass/Distinction = Favourable; Fail/Withdrawn = Unfavourable",
            "- Week-4 cutoff rule: only information from day 0 through day 27 is allowed in engineered features",
            "- Leakage exclusions: no `date_unregistration`, no clicks after day 27, no assessment score in the main model, and preprocessing must be fit on training data only",
            "- Dataset note: this workspace's `vle.csv` does not include the optional `recommended` column, so `recommended_clicks_w4` and `recommended_click_ratio_w4` are schema-preserving placeholders with zero values in this run",
            "- Recommended reuse: the group can join this table back to later-stage experiments, but should keep the same cutoff and leakage exclusions when comparing to this milestone",
        ]
    )
    path.write_text(content + "\n", encoding="utf-8")
    return path


def save_run_summary(
    config: PipelineConfig,
    row_counts: dict[str, Any],
    model_outputs: dict[str, Any],
) -> Path:
    path = config.artifacts_dir / "run_summary.md"
    calibration_delta = (
        model_outputs["calibrated_catboost_validation_log_loss"]
        - model_outputs["raw_catboost_validation_log_loss"]
    )
    lines = [
        "# Run Summary",
        "",
        "## Data Snapshot",
        "",
        f"- Students in feature table: {row_counts.get('studentInfo', 'n/a')}",
        f"- Raw `studentVle` events: {row_counts.get('studentVle', 'n/a')}",
        f"- Week-4 `studentVle` events: {row_counts.get('studentVle_week4_events', 'n/a')}",
        f"- Students with engineered week-4 click rows before left join: {row_counts.get('studentVle_filtered_feature_rows', 'n/a')}",
        f"- Early assessments due by week 4: {row_counts.get('assessments_due_by_week4', 'n/a')}",
        f"- Early assessment submissions by week 4: {row_counts.get('studentAssessment_submitted_by_week4', 'n/a')}",
        "",
        "## Model Notes",
        "",
        f"- Best validation threshold for Logistic Regression: {model_outputs['logistic_best_threshold']}",
        f"- Best validation threshold for calibrated CatBoost: {model_outputs['calibrated_best_threshold']}",
        f"- Logistic validation log loss: {model_outputs['logistic_validation_log_loss']:.6f}",
        f"- Raw CatBoost validation log loss: {model_outputs['raw_catboost_validation_log_loss']:.6f}",
        f"- Calibrated CatBoost validation log loss: {model_outputs['calibrated_catboost_validation_log_loss']:.6f}",
        f"- Calibration delta (calibrated minus raw CatBoost log loss): {calibration_delta:.6f}",
        "",
        "## Important Caveats",
        "",
        "- `vle.csv` in this workspace does not contain the optional `recommended` field from some OULAD variants, so recommended-resource features are included as zero-valued placeholders.",
        "- Sigmoid calibration improved the probability-modeling story for the report, but it did not beat raw CatBoost on validation log loss in this specific run; that should be stated honestly in the discussion.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_full_pipeline(project_root: Path | str, force_rebuild: bool = False) -> dict[str, Any]:
    config = PipelineConfig.from_project_root(project_root)
    ensure_directories(config)

    feature_table_path = config.artifacts_dir / "week4_feature_table.parquet"
    row_counts_path = config.artifacts_dir / "row_counts.csv"
    rebuild_required = force_rebuild or not feature_table_path.exists()
    row_counts: dict[str, Any]
    if not rebuild_required and row_counts_path.exists():
        row_counts_df = pd.read_csv(row_counts_path)
        row_counts = dict(zip(row_counts_df["table"], row_counts_df["row_count"]))
        if "studentVle_week4_exact_duplicates_removed" not in row_counts:
            rebuild_required = True
    else:
        row_counts = {}

    if not rebuild_required:
        feature_table = pd.read_parquet(feature_table_path)
    else:
        feature_table, row_counts = build_feature_table(config)
        feature_table.to_parquet(feature_table_path, index=False)
        save_feature_dictionary(config, feature_table)
        save_eda_outputs(config, feature_table, row_counts)

    save_feature_dictionary(config, feature_table)
    literature = save_literature_matrix(config)
    eda_outputs = save_eda_outputs(config, feature_table, row_counts)
    model_outputs = fit_and_evaluate_models(config, feature_table)
    integration_note_path = save_integration_note(config)
    run_summary_path = save_run_summary(config, row_counts, model_outputs)

    metadata = {
        "feature_table_path": feature_table_path,
        "literature_matrix_rows": len(literature),
        "row_counts": row_counts,
        "eda_outputs": eda_outputs,
        "integration_note_path": integration_note_path,
        "run_summary_path": run_summary_path,
        **model_outputs,
    }
    return metadata


def main() -> None:
    project_root = Path(__file__).resolve().parent
    results = run_full_pipeline(project_root)
    summary = {
        "feature_table_path": str(results["feature_table_path"]),
        "metrics_summary_path": str(results["metrics_summary_path"]),
        "threshold_sweep_path": str(results["threshold_sweep_path"]),
        "roc_plot_path": str(results["roc_plot_path"]),
        "threshold_plot_path": str(results["threshold_plot_path"]),
        "logistic_best_threshold": results["logistic_best_threshold"],
        "calibrated_best_threshold": results["calibrated_best_threshold"],
        "logistic_validation_log_loss": results["logistic_validation_log_loss"],
        "raw_catboost_validation_log_loss": results["raw_catboost_validation_log_loss"],
        "calibrated_catboost_validation_log_loss": results[
            "calibrated_catboost_validation_log_loss"
        ],
    }
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()
