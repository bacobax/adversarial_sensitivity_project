"""
Analysis Runner Script for Forgery-Localization Pipeline

This script performs quantitative analysis and visualization on the output
of spatial_corr_test.py. It loads CSV files from the analysis/ directory,
computes aggregated metrics, and generates charts and tables.

Input structure (from spatial_corr_test.py):
    analysis/
    ├── <model>/
    │   ├── metrics_explanation.csv          # Attack-independent explanation metrics
    │   ├── <attack>/
    │   │   └── metrics_vulnerability.csv    # Per-attack vulnerability metrics
    │   └── vis/
    │       └── <attack>/
    │           └── <filename>_grid.png      # Visualization grids

Output structure:
    analysis_results/
    ├── tables/
    │   ├── aggregated_explanation.csv
    │   ├── aggregated_vulnerability.csv
    │   ├── explanation_samecat_vs_diffcat.csv
    │   └── vulnerability_samecat_vs_diffcat.csv
    └── charts/
        ├── explanation/
        │   ├── mass_frac_bar.png            # Bar chart by model and image_type
        │   ├── pr_auc_heatmap.png           # Heatmap by model (attack-independent)
        │   ├── samecat_vs_diffcat_scatter.png
        │   └── mass_frac_distribution.png
        └── vulnerability/
            ├── mass_frac_bar.png            # Bar chart by model and attack_type
            ├── pr_auc_heatmap.png           # Heatmap by model × attack_type
            ├── samecat_vs_diffcat_scatter.png
            └── mass_frac_distribution.png
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# =============================================================================
# Configuration
# =============================================================================

# Base directory (where this script is located)
BASE_DIR = Path(__file__).parent.resolve()

# Default input and output directories
DEFAULT_ANALYSIS_DIR = BASE_DIR / "analysis"
DEFAULT_OUTPUT_DIR = BASE_DIR / "analysis_results"

# Module-level directories (will be set in main based on args)
ANALYSIS_DIR = DEFAULT_ANALYSIS_DIR
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
TABLES_DIR = OUTPUT_DIR / "tables"
CHARTS_DIR = OUTPUT_DIR / "charts"
EXPLANATION_CHARTS_DIR = CHARTS_DIR / "explanation"
VULNERABILITY_CHARTS_DIR = CHARTS_DIR / "vulnerability"

# Metrics to analyze
# Include IoU (stored in CSV as `iou_topk`) as requested
MAIN_METRICS = ["iou_topk", "mass_frac", "pr_auc", "roc_auc"]

# Friendly display names for metrics
METRIC_DISPLAY_NAMES = {
    "iou_topk": "IoU",
    "mass_frac": "Mass Fraction",
    "pr_auc": "PR-AUC",
    "roc_auc": "ROC-AUC",
}

# Plot style configuration
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
FIGURE_DPI = 150
FIGURE_SIZE = (10, 6)
HEATMAP_SIZE = (12, 8)


# =============================================================================
# Directory Setup
# =============================================================================

def create_output_directories() -> None:
    """Create all required output directories if they don't exist."""
    directories = [
        OUTPUT_DIR,
        TABLES_DIR,
        CHARTS_DIR,
        EXPLANATION_CHARTS_DIR,
        VULNERABILITY_CHARTS_DIR,
        CHARTS_DIR / "exp_vs_vuln",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directories created at: {OUTPUT_DIR}")


# =============================================================================
# Data Loading Functions
# =============================================================================

def discover_analysis_structure(analysis_dir: Path) -> Tuple[List[Tuple[str, Path]], List[Tuple[str, str, Path]]]:
    """
    Walk the analysis directory and discover model/attack structure.
    
    New structure:
      - Explanation CSV: [model]/metrics_explanation.csv (attack-independent)
      - Vulnerability CSV: [model]/[attack_type]/metrics_vulnerability.csv (per attack)
    
    Returns:
        Tuple of:
          - List of tuples: (model_name, explanation_csv_path)
          - List of tuples: (model_name, attack_type, vulnerability_csv_path)
    """
    explanation_discoveries = []
    vulnerability_discoveries = []
    
    if not analysis_dir.exists():
        raise FileNotFoundError(f"Analysis directory not found: {analysis_dir}")
    
    # Walk through MODEL_NAME directories
    for model_dir in analysis_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        
        # Check for explanation CSV at model level (attack-independent)
        explanation_csv = model_dir / "metrics_explanation.csv"
        if explanation_csv.exists():
            explanation_discoveries.append((model_name, explanation_csv))
        
        # Walk through ATTACK_TYPE directories for vulnerability CSVs
        for attack_dir in model_dir.iterdir():
            if not attack_dir.is_dir():
                continue
            
            attack_type = attack_dir.name
            
            vulnerability_csv = attack_dir / "metrics_vulnerability.csv"
            
            if vulnerability_csv.exists():
                vulnerability_discoveries.append((
                    model_name,
                    attack_type,
                    vulnerability_csv
                ))
    
    return explanation_discoveries, vulnerability_discoveries


def load_csv_with_metadata(csv_path: Path, model_name: str, attack_type: str = None) -> pd.DataFrame:
    """
    Load a CSV file and add model/attack metadata columns.
    
    Args:
        csv_path: Path to the CSV file
        model_name: Name of the model (detector)
        attack_type: Type of attack (optional, only for vulnerability CSVs)
    
    Returns:
        DataFrame with added metadata columns
    """
    if csv_path is None or not csv_path.exists():
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(csv_path)
        df["model"] = model_name
        # Only add attack_type for vulnerability data
        if attack_type is not None:
            if "attack_type" not in df.columns:
                df["attack_type"] = attack_type
        return df
    except Exception as e:
        print(f"  Warning: Could not load {csv_path}: {e}")
        return pd.DataFrame()


def load_all_data(analysis_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load all CSV files from the analysis directory into two DataFrames.
    
    New structure:
      - Explanation: [model]/metrics_explanation.csv (attack-independent)
      - Vulnerability: [model]/[attack_type]/metrics_vulnerability.csv
    
    Returns:
        Tuple of (explanation_df, vulnerability_df)
    """
    print("Loading data from analysis directory...")
    
    explanation_discoveries, vulnerability_discoveries = discover_analysis_structure(analysis_dir)
    
    explanation_dfs = []
    vulnerability_dfs = []
    
    # Load explanation CSVs (attack-independent, at model level)
    for model_name, exp_path in explanation_discoveries:
        print(f"  Loading explanation: {model_name}")
        exp_df = load_csv_with_metadata(exp_path, model_name, attack_type=None)
        if not exp_df.empty:
            explanation_dfs.append(exp_df)
    
    # Load vulnerability CSVs (per attack)
    for model_name, attack_type, vuln_path in vulnerability_discoveries:
        print(f"  Loading vulnerability: {model_name}/{attack_type}")
        vuln_df = load_csv_with_metadata(vuln_path, model_name, attack_type)
        if not vuln_df.empty:
            vulnerability_dfs.append(vuln_df)
    
    # Concatenate all DataFrames
    explanation_df = pd.concat(explanation_dfs, ignore_index=True) if explanation_dfs else pd.DataFrame()
    vulnerability_df = pd.concat(vulnerability_dfs, ignore_index=True) if vulnerability_dfs else pd.DataFrame()
    
    print(f"✓ Loaded {len(explanation_df)} explanation records")
    print(f"✓ Loaded {len(vulnerability_df)} vulnerability records")
    
    return explanation_df, vulnerability_df


# =============================================================================
# Aggregation Functions
# =============================================================================

def compute_aggregated_metrics(
    df: pd.DataFrame,
    group_cols: List[str],
    metrics: List[str] = MAIN_METRICS
) -> pd.DataFrame:
    """
    Compute mean metrics grouped by specified columns.
    
    Args:
        df: Input DataFrame
        group_cols: Columns to group by
        metrics: Metrics to aggregate
    
    Returns:
        Aggregated DataFrame with mean values
    """
    if df.empty:
        return pd.DataFrame()
    
    # Filter to only existing metrics
    existing_metrics = [m for m in metrics if m in df.columns]
    
    if not existing_metrics:
        return pd.DataFrame()
    
    # Compute aggregation
    agg_dict = {metric: "mean" for metric in existing_metrics}
    aggregated = df.groupby(group_cols, as_index=False).agg(agg_dict)
    
    # Rename columns to indicate they are means
    rename_dict = {metric: f"mean_{metric}" for metric in existing_metrics}
    aggregated = aggregated.rename(columns=rename_dict)
    
    return aggregated


def compute_aggregated_metrics_transposed(
    df: pd.DataFrame,
    metrics: List[str] = MAIN_METRICS,
    include_attack: bool = False
) -> pd.DataFrame:
    """
    Compute mean metrics with metrics as rows and models (optionally with attack) as columns.
    When include_attack=True, produces multi-level column headers.
    
    Args:
        df: Input DataFrame
        metrics: Metrics to aggregate
        include_attack: If True, creates two-row header (model, attack_type)
    
    Returns:
        Transposed DataFrame: rows = metrics, columns = models (or model_attack with multi-level headers)
    """
    if df.empty:
        return pd.DataFrame()
    
    # Filter to only existing metrics
    existing_metrics = [m for m in metrics if m in df.columns]
    
    if not existing_metrics:
        return pd.DataFrame()
    
    if include_attack and "attack_type" in df.columns:
        # Group by model and attack_type
        grouped = df.groupby(["model", "attack_type"], as_index=False)[existing_metrics].mean()
        
        # Melt to long format
        melted = grouped.melt(id_vars=["model", "attack_type"], value_vars=existing_metrics,
                              var_name="metric", value_name="value")
        
        # Pivot to create multi-level columns
        pivoted = melted.pivot(index="metric", columns=["model", "attack_type"], values="value")
        pivoted = pivoted.reset_index()
        pivoted = pivoted.rename(columns={"metric": "Metric"})
        
        # Create two-row header manually for CSV export
        # We'll construct a special structure that save_tables_with_multiindex will handle
        pivoted.attrs["multi_level"] = True
        pivoted.attrs["level_names"] = ["Model", "Attack"]
        
        return pivoted
    else:
        # Group by model only
        grouped = df.groupby("model", as_index=False)[existing_metrics].mean()
        
        # Melt to long format
        melted = grouped.melt(id_vars=["model"], value_vars=existing_metrics,
                              var_name="metric", value_name="value")
        
        # Pivot to get model as columns
        pivoted = melted.pivot(index="metric", columns="model", values="value")
        pivoted = pivoted.reset_index()
        pivoted = pivoted.rename(columns={"metric": "Metric"})
        
        return pivoted


def compute_samecat_vs_diffcat_table(
    df: pd.DataFrame,
    metrics: List[str] = MAIN_METRICS
) -> pd.DataFrame:
    """
    Create a comparison table of samecat vs diffcat metrics per model.
    Format: Two columns (Metric, Image_Type) as row identifiers, then model columns.
    
    Returns:
        DataFrame with two first columns: Metric and Image_Type,
        followed by model columns with values
    """
    if df.empty or "image_type" not in df.columns:
        return pd.DataFrame()
    
    # Filter to only existing metrics
    existing_metrics = [m for m in metrics if m in df.columns]
    
    if not existing_metrics:
        return pd.DataFrame()
    
    # Compute per-model, per-image_type means
    grouped = df.groupby(["model", "image_type"], as_index=False)[existing_metrics].mean()
    
    # Melt to long format
    melted = grouped.melt(id_vars=["model", "image_type"], value_vars=existing_metrics,
                          var_name="metric", value_name="value")
    
    # Pivot: (metric, image_type) as index, model as columns
    pivoted = melted.pivot(index=["metric", "image_type"], columns="model", values="value")
    pivoted = pivoted.reset_index()
    
    # Rename columns
    pivoted = pivoted.rename(columns={"metric": "Metric", "image_type": "Image_Type"})
    
    # Sort for readability: first by metric, then by image_type
    pivoted = pivoted.sort_values(["Metric", "Image_Type"]).reset_index(drop=True)
    
    return pivoted


def generate_all_aggregations(
    explanation_df: pd.DataFrame,
    vulnerability_df: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """
    Generate all required aggregation tables.
    Tables are transposed: metrics as rows, models (or model_attack) as columns.
    
    Returns:
        Dictionary mapping table names to DataFrames
    """
    print("Computing aggregations...")
    
    tables = {}
    
    # === Explanation Metrics ===
    # Transposed table: metrics as rows, models as columns
    # Include attack in column names if attack_type exists
    has_attack = "attack_type" in explanation_df.columns if not explanation_df.empty else False
    exp_aggregated = compute_aggregated_metrics_transposed(
        explanation_df, 
        include_attack=has_attack
    )
    if not exp_aggregated.empty:
        tables["aggregated_explanation"] = exp_aggregated
    
    # samecat vs diffcat comparison (transposed: metric_imagetype as rows, models as columns)
    exp_samecat_diffcat = compute_samecat_vs_diffcat_table(explanation_df)
    if not exp_samecat_diffcat.empty:
        tables["explanation_samecat_vs_diffcat"] = exp_samecat_diffcat
    
    # === Vulnerability Metrics ===
    # Transposed table: metrics as rows, model_attack as columns
    vuln_aggregated = compute_aggregated_metrics_transposed(
        vulnerability_df,
        include_attack=True
    )
    if not vuln_aggregated.empty:
        tables["aggregated_vulnerability"] = vuln_aggregated
    
    # samecat vs diffcat comparison for vulnerability (transposed)
    vuln_samecat_diffcat = compute_samecat_vs_diffcat_table(vulnerability_df)
    if not vuln_samecat_diffcat.empty:
        tables["vulnerability_samecat_vs_diffcat"] = vuln_samecat_diffcat
    
    print(f"✓ Generated {len(tables)} aggregation tables")
    
    return tables


def save_tables(tables: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Save all aggregation tables as CSV files, handling multi-level headers."""
    print("Saving tables...")
    
    for name, df in tables.items():
        if df.empty:
            continue
        
        output_path = output_dir / f"{name}.csv"
        
        # Check if this table has multi-level column headers
        if hasattr(df, 'attrs') and df.attrs.get('multi_level', False):
            # Save with multi-level headers
            # First, we need to construct the multi-level structure
            if isinstance(df.columns, pd.MultiIndex):
                # Already has MultiIndex, save directly
                df.to_csv(output_path, index=False)
            else:
                # Need to reconstruct as MultiIndex for proper CSV output
                # The first column is 'Metric', rest are (model, attack) tuples
                cols = df.columns.tolist()
                if cols[0] == 'Metric':
                    # Create new column structure
                    new_cols = [('', 'Metric')]  # First column has empty top level
                    for col in cols[1:]:
                        if isinstance(col, tuple) and len(col) == 2:
                            new_cols.append(col)
                        else:
                            # Shouldn't happen, but handle gracefully
                            new_cols.append(('', str(col)))
                    
                    df.columns = pd.MultiIndex.from_tuples(new_cols)
                    df.to_csv(output_path, index=False)
                else:
                    df.to_csv(output_path, index=False)
        else:
            df.to_csv(output_path, index=False)
        
        print(f"  Saved: {output_path.name}")
    
    print(f"✓ Tables saved to: {output_dir}")


# =============================================================================
# Plotting Functions - Explanation Track
# =============================================================================

def plot_explanation_mass_frac_bar(
    df: pd.DataFrame, output_path: Path, ylim: tuple[float, float] | None = None
) -> None:
    """
    E1: Bar chart of mass_frac per model, grouped by image_type.
    """
    if df.empty or "mass_frac" not in df.columns or "image_type" not in df.columns:
        print("  Skipping E1: insufficient data")
        return
    
    # Compute means
    grouped = df.groupby(["model", "image_type"], as_index=False)["mass_frac"].mean()
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    # Create grouped bar chart
    models = grouped["model"].unique()
    image_types = grouped["image_type"].unique()
    x = np.arange(len(models))
    width = 0.35
    
    colors = sns.color_palette("husl", len(image_types))
    
    for i, img_type in enumerate(image_types):
        subset = grouped[grouped["image_type"] == img_type]
        # Align data with model order
        values = []
        for model in models:
            val = subset[subset["model"] == model]["mass_frac"].values
            values.append(val[0] if len(val) > 0 else 0)
        
        offset = (i - len(image_types)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=img_type, color=colors[i])
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Mean Mass Fraction", fontsize=12)
    ax.set_title("Explanation Maps: Mass Fraction by Model and Image Type", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend(title="Image Type")
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    else:
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_explanation_pr_auc_heatmap(
    df: pd.DataFrame, output_path: Path, vlim: tuple[float, float] | None = None
) -> None:
    """
    E2: Heatmap of PR-AUC for each (model × image_type).
    Explanation is attack-independent, so we show metrics by image_type.
    """
    if df.empty or "pr_auc" not in df.columns or "image_type" not in df.columns:
        print("  Skipping E2: insufficient data")
        return
    
    # Compute means by model and image_type
    grouped = df.groupby(["model", "image_type"], as_index=False)["pr_auc"].mean()
    
    # Pivot for heatmap
    pivot_df = grouped.pivot(index="model", columns="image_type", values="pr_auc")
    
    fig, ax = plt.subplots(figsize=HEATMAP_SIZE)
    
    vmin_val = vlim[0] if vlim is not None else 0
    vmax_val = vlim[1] if vlim is not None else 1
    
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        ax=ax,
        vmin=vmin_val,
        vmax=vmax_val,
        cbar_kws={"label": "Mean PR-AUC"},
        linewidths=0.5
    )

    ax.set_title("Explanation Maps: PR-AUC by Model and Image Type", fontsize=14)
    ax.set_xlabel("Image Type", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_explanation_samecat_vs_diffcat_scatter(
    df: pd.DataFrame,
    output_path: Path,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """
    E3: Scatter plot of mean_mass_frac(samecat) vs mean_mass_frac(diffcat) per model.
    """
    if df.empty or "mass_frac" not in df.columns or "image_type" not in df.columns:
        print("  Skipping E3: insufficient data")
        return
    
    # Compute per-model, per-image_type means
    grouped = df.groupby(["model", "image_type"], as_index=False)["mass_frac"].mean()
    
    # Pivot to get samecat and diffcat columns
    pivot_df = grouped.pivot(index="model", columns="image_type", values="mass_frac")
    pivot_df = pivot_df.reset_index()
    
    if "samecat" not in pivot_df.columns or "diffcat" not in pivot_df.columns:
        print("  Skipping E3: missing samecat or diffcat data")
        return
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    colors = sns.color_palette("husl", len(pivot_df))
    
    for i, (_, row) in enumerate(pivot_df.iterrows()):
        ax.scatter(row["samecat"], row["diffcat"], s=150, c=[colors[i]], 
                   label=row["model"], edgecolors="black", linewidth=1)
        ax.annotate(row["model"], (row["samecat"], row["diffcat"]),
                   textcoords="offset points", xytext=(5, 5), fontsize=9)
    
    # Use provided limits or compute optimized square range from data
    if xlim is not None and ylim is not None:
        axis_min, axis_max = xlim[0], xlim[1]
    else:
        # Compute square range that encompasses all points with padding
        all_x = pivot_df["samecat"].values
        all_y = pivot_df["diffcat"].values
        data_min = min(all_x.min(), all_y.min())
        data_max = max(all_x.max(), all_y.max())
        data_range = data_max - data_min
        padding = max(0.05, data_range * 0.15)  # 15% padding or at least 0.05
        axis_min = max(0, data_min - padding)
        axis_max = min(1, data_max + padding)
    
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    
    # Add diagonal reference line
    ax.plot([axis_min, axis_max], [axis_min, axis_max], 'k--', alpha=0.5, label='x=y')
    
    ax.set_xlabel("Mean Mass Fraction (samecat)", fontsize=12)
    ax.set_ylabel("Mean Mass Fraction (diffcat)", fontsize=12)
    ax.set_title("Explanation Maps: Samecat vs Diffcat Comparison", fontsize=14)
    ax.legend(loc="best", fontsize=8)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_explanation_mass_frac_distribution(
    df: pd.DataFrame, output_path: Path, xlim: tuple[float, float] | None = None
) -> None:
    """
    E4: Distribution (KDE) of mass_frac per image, overlayed per model.
    Explanation data is attack-independent (computed from original benign images).
    """
    if df.empty or "mass_frac" not in df.columns:
        print("  Skipping E4: insufficient data")
        return
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    models = df["model"].unique()
    colors = sns.color_palette("husl", len(models))
    
    for i, model in enumerate(models):
        model_data = df[df["model"] == model]["mass_frac"].dropna()
        if len(model_data) > 1:
            sns.kdeplot(data=model_data, ax=ax, label=model, color=colors[i], linewidth=2)
    
    ax.set_xlabel("Mass Fraction", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Explanation Maps: Mass Fraction Distribution by Model", fontsize=14)
    ax.legend(title="Model", loc="best")
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    else:
        ax.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


# =============================================================================
# Plotting Functions - Vulnerability Track
# =============================================================================

def plot_vulnerability_mass_frac_bar(
    df: pd.DataFrame, output_path: Path, ylim: tuple[float, float] | None = None
) -> None:
    """
    V1: Bar chart of mass_frac_vuln per model, grouped by attack_type.
    """
    if df.empty or "mass_frac" not in df.columns or "attack_type" not in df.columns:
        print("  Skipping V1: insufficient data")
        return
    
    # Compute means
    grouped = df.groupby(["model", "attack_type"], as_index=False)["mass_frac"].mean()
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    models = grouped["model"].unique()
    attack_types = grouped["attack_type"].unique()
    x = np.arange(len(models))
    width = 0.25
    
    colors = sns.color_palette("husl", len(attack_types))
    
    for i, attack in enumerate(attack_types):
        subset = grouped[grouped["attack_type"] == attack]
        values = []
        for model in models:
            val = subset[subset["model"] == model]["mass_frac"].values
            values.append(val[0] if len(val) > 0 else 0)
        
        offset = (i - len(attack_types)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=attack, color=colors[i])
        
        # Add value labels
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=90)
    
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Mean Mass Fraction (Vulnerability)", fontsize=12)
    ax.set_title("Vulnerability Maps: Mass Fraction by Model and Attack Type", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend(title="Attack Type")
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    else:
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_vulnerability_pr_auc_heatmap(
    df: pd.DataFrame, output_path: Path, vlim: tuple[float, float] | None = None
) -> None:
    """
    V2: Heatmap of PR-AUC for vulnerability maps (model × attack_type).
    """
    if df.empty or "pr_auc" not in df.columns or "attack_type" not in df.columns:
        print("  Skipping V2: insufficient data")
        return
    
    # Compute means
    grouped = df.groupby(["model", "attack_type"], as_index=False)["pr_auc"].mean()
    
    # Pivot for heatmap
    pivot_df = grouped.pivot(index="model", columns="attack_type", values="pr_auc")
    
    fig, ax = plt.subplots(figsize=HEATMAP_SIZE)
    
    vmin_val = vlim[0] if vlim is not None else 0
    vmax_val = vlim[1] if vlim is not None else 1
    
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        ax=ax,
        vmin=vmin_val,
        vmax=vmax_val,
        cbar_kws={"label": "Mean PR-AUC"},
        linewidths=0.5
    )

    ax.set_title("Vulnerability Maps: PR-AUC by Model and Attack Type", fontsize=14)
    ax.set_xlabel("Attack Type", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_vulnerability_samecat_vs_diffcat_scatter(
    df: pd.DataFrame,
    output_path: Path,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """
    V3: Scatter plot of mean_mass_frac_vuln(samecat) vs mean_mass_frac_vuln(diffcat).
    Points colored by attack_type.
    """
    if df.empty or "mass_frac" not in df.columns or "image_type" not in df.columns:
        print("  Skipping V3: insufficient data")
        return
    
    if "attack_type" not in df.columns:
        print("  Skipping V3: no attack_type column")
        return
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    attack_types = df["attack_type"].unique()
    models = df["model"].unique()
    colors = sns.color_palette("husl", len(attack_types))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    
    # Collect all points for axis range calculation
    all_samecat = []
    all_diffcat = []
    
    for i, attack in enumerate(attack_types):
        attack_df = df[df["attack_type"] == attack]
        
        # Compute per-model, per-image_type means for this attack
        grouped = attack_df.groupby(["model", "image_type"], as_index=False)["mass_frac"].mean()
        
        # Pivot
        pivot_df = grouped.pivot(index="model", columns="image_type", values="mass_frac")
        pivot_df = pivot_df.reset_index()
        
        if "samecat" not in pivot_df.columns or "diffcat" not in pivot_df.columns:
            continue
        
        all_samecat.extend(pivot_df["samecat"].values)
        all_diffcat.extend(pivot_df["diffcat"].values)
        
        marker = markers[i % len(markers)]
        
        for _, row in pivot_df.iterrows():
            ax.scatter(row["samecat"], row["diffcat"], s=100, c=[colors[i]],
                      marker=marker, edgecolors="black", linewidth=0.5,
                      label=f"{attack}" if row.name == 0 else "")
            ax.annotate(f"{row['model']}", (row["samecat"], row["diffcat"]),
                       textcoords="offset points", xytext=(3, 3), fontsize=7)
    
    # Use provided limits or compute optimized square range from data
    if xlim is not None and ylim is not None:
        axis_min, axis_max = xlim[0], xlim[1]
    elif all_samecat and all_diffcat:
        # Compute square range that encompasses all points with padding
        data_min = min(min(all_samecat), min(all_diffcat))
        data_max = max(max(all_samecat), max(all_diffcat))
        data_range = data_max - data_min
        padding = max(0.05, data_range * 0.15)  # 15% padding or at least 0.05
        axis_min = max(0, data_min - padding)
        axis_max = min(1, data_max + padding)
    else:
        axis_min, axis_max = 0, 1
    
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    
    # Add diagonal reference line
    ax.plot([axis_min, axis_max], [axis_min, axis_max], 'k--', alpha=0.5, label='x=y')
    
    ax.set_xlabel("Mean Mass Fraction (samecat)", fontsize=12)
    ax.set_ylabel("Mean Mass Fraction (diffcat)", fontsize=12)
    ax.set_title("Vulnerability Maps: Samecat vs Diffcat by Attack Type", fontsize=14)
    
    # Create legend with unique entries
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="best", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_vulnerability_mass_frac_distribution(
    df: pd.DataFrame, output_path: Path, xlim: tuple[float, float] | None = None
) -> None:
    """
    V4: Distribution (KDE) of mass_frac_vuln across images, per attack_type.
    """
    if df.empty or "mass_frac" not in df.columns:
        print("  Skipping V4: insufficient data")
        return
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    if "attack_type" in df.columns:
        attack_types = df["attack_type"].unique()
        colors = sns.color_palette("husl", len(attack_types))
        
        for i, attack in enumerate(attack_types):
            attack_data = df[df["attack_type"] == attack]["mass_frac"].dropna()
            if len(attack_data) > 1:
                sns.kdeplot(data=attack_data, ax=ax, label=attack, 
                           color=colors[i], linewidth=2)
        
        ax.legend(title="Attack Type", loc="best")
    else:
        # Fallback: plot by model
        models = df["model"].unique()
        colors = sns.color_palette("husl", len(models))
        
        for i, model in enumerate(models):
            model_data = df[df["model"] == model]["mass_frac"].dropna()
            if len(model_data) > 1:
                sns.kdeplot(data=model_data, ax=ax, label=model,
                           color=colors[i], linewidth=2)
        
        ax.legend(title="Model", loc="best")
    
    ax.set_xlabel("Mass Fraction (Vulnerability)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Vulnerability Maps: Mass Fraction Distribution", fontsize=14)
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    else:
        ax.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


# =============================================================================
# Chart Generation Orchestration
# =============================================================================

def compute_shared_ranges(
    explanation_df: pd.DataFrame, vulnerability_df: pd.DataFrame
) -> dict:
    """
    Compute shared axis ranges for paired plots across explanation and vulnerability data.
    Returns a dict with keys for each plot type.
    """
    ranges = {}
    
    # mass_frac_bar: y-axis is mean mass_frac (computed from grouped means)
    all_mass_frac_means = []
    if not explanation_df.empty and "mass_frac" in explanation_df.columns:
        if "image_type" in explanation_df.columns:
            grouped = explanation_df.groupby(["model", "image_type"], as_index=False)["mass_frac"].mean()
            all_mass_frac_means.extend(grouped["mass_frac"].dropna().tolist())
    if not vulnerability_df.empty and "mass_frac" in vulnerability_df.columns:
        if "attack_type" in vulnerability_df.columns:
            grouped = vulnerability_df.groupby(["model", "attack_type"], as_index=False)["mass_frac"].mean()
            all_mass_frac_means.extend(grouped["mass_frac"].dropna().tolist())
    
    if all_mass_frac_means:
        bar_min = min(all_mass_frac_means)
        bar_max = max(all_mass_frac_means)
        bar_padding = max(0.02, (bar_max - bar_min) * 0.1)
        ranges["mass_frac_bar_ylim"] = (max(0, bar_min - bar_padding), min(1, bar_max + bar_padding))
    else:
        ranges["mass_frac_bar_ylim"] = (0, 1)
    
    # pr_auc_heatmap: vmin/vmax from grouped mean PR-AUC values (same as what's displayed)
    all_pr_auc_means = []
    if not explanation_df.empty and "pr_auc" in explanation_df.columns:
        if "image_type" in explanation_df.columns:
            grouped = explanation_df.groupby(["model", "image_type"], as_index=False)["pr_auc"].mean()
            all_pr_auc_means.extend(grouped["pr_auc"].dropna().tolist())
    if not vulnerability_df.empty and "pr_auc" in vulnerability_df.columns:
        if "attack_type" in vulnerability_df.columns:
            grouped = vulnerability_df.groupby(["model", "attack_type"], as_index=False)["pr_auc"].mean()
            all_pr_auc_means.extend(grouped["pr_auc"].dropna().tolist())
    
    if all_pr_auc_means:
        pr_min = min(all_pr_auc_means)
        pr_max = max(all_pr_auc_means)
        pr_padding = max(0.02, (pr_max - pr_min) * 0.1)
        ranges["pr_auc_vlim"] = (max(0, pr_min - pr_padding), min(1, pr_max + pr_padding))
    else:
        ranges["pr_auc_vlim"] = (0, 1)
    
    # samecat_vs_diffcat: xlim/ylim from pivot means
    all_samecat = []
    all_diffcat = []
    
    for df in [explanation_df, vulnerability_df]:
        if df.empty or "mass_frac" not in df.columns or "image_type" not in df.columns:
            continue
        
        # Handle per-attack grouping for vulnerability
        if "attack_type" in df.columns:
            for attack in df["attack_type"].unique():
                attack_df = df[df["attack_type"] == attack]
                grouped = attack_df.groupby(["model", "image_type"], as_index=False)["mass_frac"].mean()
                pivot = grouped.pivot(index="model", columns="image_type", values="mass_frac")
                if "samecat" in pivot.columns:
                    all_samecat.extend(pivot["samecat"].dropna().tolist())
                if "diffcat" in pivot.columns:
                    all_diffcat.extend(pivot["diffcat"].dropna().tolist())
        else:
            grouped = df.groupby(["model", "image_type"], as_index=False)["mass_frac"].mean()
            pivot = grouped.pivot(index="model", columns="image_type", values="mass_frac")
            if "samecat" in pivot.columns:
                all_samecat.extend(pivot["samecat"].dropna().tolist())
            if "diffcat" in pivot.columns:
                all_diffcat.extend(pivot["diffcat"].dropna().tolist())
    
    if all_samecat and all_diffcat:
        x_min, x_max = min(all_samecat), max(all_samecat)
        y_min, y_max = min(all_diffcat), max(all_diffcat)
        x_padding = max(0.05, (x_max - x_min) * 0.15)
        y_padding = max(0.05, (y_max - y_min) * 0.15)
        ranges["scatter_xlim"] = (max(0, x_min - x_padding), min(1, x_max + x_padding))
        ranges["scatter_ylim"] = (max(0, y_min - y_padding), min(1, y_max + y_padding))
    else:
        ranges["scatter_xlim"] = (0, 1)
        ranges["scatter_ylim"] = (0, 1)
    
    # mass_frac_distribution: xlim from all mass_frac values
    all_mass_frac = []
    if not explanation_df.empty and "mass_frac" in explanation_df.columns:
        all_mass_frac.extend(explanation_df["mass_frac"].dropna().tolist())
    if not vulnerability_df.empty and "mass_frac" in vulnerability_df.columns:
        all_mass_frac.extend(vulnerability_df["mass_frac"].dropna().tolist())
    
    if all_mass_frac:
        dist_min = min(all_mass_frac)
        dist_max = max(all_mass_frac)
        dist_padding = max(0.02, (dist_max - dist_min) * 0.1)
        ranges["distribution_xlim"] = (max(0, dist_min - dist_padding), min(1, dist_max + dist_padding))
    else:
        ranges["distribution_xlim"] = (0, 1)
    
    # Debug: print computed ranges
    print("  Computed shared ranges:")
    for key, val in ranges.items():
        print(f"    {key}: {val}")
    
    return ranges


def generate_explanation_charts(
    df: pd.DataFrame, output_dir: Path, shared_ranges: dict | None = None
) -> None:
    """Generate all explanation track charts."""
    print("Generating explanation charts...")
    
    if shared_ranges is None:
        shared_ranges = {}
    
    plot_explanation_mass_frac_bar(
        df, output_dir / "mass_frac_bar.png",
        ylim=shared_ranges.get("mass_frac_bar_ylim")
    )
    plot_explanation_pr_auc_heatmap(
        df, output_dir / "pr_auc_heatmap.png",
        vlim=shared_ranges.get("pr_auc_vlim")
    )
    plot_explanation_samecat_vs_diffcat_scatter(
        df, output_dir / "samecat_vs_diffcat_scatter.png"
    )
    plot_explanation_mass_frac_distribution(
        df, output_dir / "mass_frac_distribution.png",
        xlim=shared_ranges.get("distribution_xlim")
    )
    
    print(f"✓ Explanation charts saved to: {output_dir}")


def generate_vulnerability_charts(
    df: pd.DataFrame, output_dir: Path, shared_ranges: dict | None = None
) -> None:
    """Generate all vulnerability track charts."""
    print("Generating vulnerability charts...")
    
    if shared_ranges is None:
        shared_ranges = {}
    
    plot_vulnerability_mass_frac_bar(
        df, output_dir / "mass_frac_bar.png",
        ylim=shared_ranges.get("mass_frac_bar_ylim")
    )
    plot_vulnerability_pr_auc_heatmap(
        df, output_dir / "pr_auc_heatmap.png",
        vlim=shared_ranges.get("pr_auc_vlim")
    )
    plot_vulnerability_samecat_vs_diffcat_scatter(
        df, output_dir / "samecat_vs_diffcat_scatter.png"
    )
    plot_vulnerability_mass_frac_distribution(
        df, output_dir / "mass_frac_distribution.png",
        xlim=shared_ranges.get("distribution_xlim")
    )
    
    print(f"✓ Vulnerability charts saved to: {output_dir}")


def plot_exp_vs_vuln_grid(
    metric: str,
    explanation_df: pd.DataFrame,
    vulnerability_df: pd.DataFrame,
    output_path: Path,
    attack_name: str = "pgd",
) -> None:
    """
    Create a grid of histograms comparing explanation vs vulnerability (fixed attack) for
    a given metric.

    Rows: models
    Columns: ['samecat', 'diffcat']

    Each cell overlays the histogram of the selected metric for explanation maps
    (from `explanation_df`) and vulnerability maps filtered to `attack_name` (from `vulnerability_df`).
    """
    if explanation_df.empty and vulnerability_df.empty:
        print(f"  Skipping exp_vs_vuln for {metric}: no data")
        return

    models = sorted(set(explanation_df.get("model", pd.Series(dtype=str)).unique())
                    | set(vulnerability_df.get("model", pd.Series(dtype=str)).unique()))
    if not models:
        print(f"  Skipping exp_vs_vuln for {metric}: no models found")
        return

    image_types = ["samecat", "diffcat"]

    nrows = len(models)
    ncols = len(image_types)

    # Figure size: width fixed, height scales with number of models
    fig_w = 10
    fig_h = max(3, nrows * 2.2)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), squeeze=False)

    colors = sns.color_palette("husl", 2)

    for i, model in enumerate(models):
        for j, img_type in enumerate(image_types):
            ax = axes[i][j]

            # Explanation values (attack-independent)
            exp_vals = pd.Series(dtype=float)
            if not explanation_df.empty:
                exp_vals = explanation_df[
                    (explanation_df["model"] == model) & (explanation_df["image_type"] == img_type)
                ][metric].dropna()

            # Vulnerability values filtered to attack (e.g., pgd)
            vuln_vals = pd.Series(dtype=float)
            if not vulnerability_df.empty and "attack_type" in vulnerability_df.columns:
                vuln_vals = vulnerability_df[
                    (vulnerability_df["model"] == model) &
                    (vulnerability_df["attack_type"] == attack_name) &
                    (vulnerability_df["image_type"] == img_type)
                ][metric].dropna()

            # If both empty, show message
            if exp_vals.empty and vuln_vals.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=10, color="#666")
                ax.set_xticks([])
                ax.set_yticks([])
                if j == 0:
                    ax.set_ylabel(model, fontsize=9)
                continue

            # Determine combined bin edges
            combined = pd.concat([exp_vals, vuln_vals]) if (not exp_vals.empty or not vuln_vals.empty) else pd.Series([])
            if combined.empty:
                bins = 10
            else:
                try:
                    vmin = float(combined.min())
                    vmax = float(combined.max())
                    if vmin == vmax:
                        bins = np.linspace(vmin - 0.001, vmax + 0.001, 10)
                    else:
                        bins = np.linspace(vmin, vmax, 20)
                except Exception:
                    bins = 10

            # Plot histograms (density normalized)
            if not exp_vals.empty:
                ax.hist(exp_vals, bins=bins, density=True, alpha=0.6, label="explain", color=colors[0], edgecolor="black")
            if not vuln_vals.empty:
                ax.hist(vuln_vals, bins=bins, density=True, alpha=0.6, label=f"vuln ({attack_name})", color=colors[1], edgecolor="black")

            # Row label
            if j == 0:
                ax.set_ylabel(model, fontsize=9)

            # Only top row gets column title
            if i == 0:
                ax.set_title(img_type, fontsize=10)

            # Tidy-up
            ax.tick_params(axis='both', which='major', labelsize=8)
            if i == nrows - 1:
                ax.set_xlabel(METRIC_DISPLAY_NAMES.get(metric, metric), fontsize=9)

            # Add legend to first row, second column to avoid repeating
            if i == 0 and j == ncols - 1:
                ax.legend(fontsize=8)

    plt.suptitle(f"Explanation vs Vulnerability ({METRIC_DISPLAY_NAMES.get(metric, metric)})", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def generate_exp_vs_vuln_charts(
    explanation_df: pd.DataFrame,
    vulnerability_df: pd.DataFrame,
    output_dir: Path,
    metrics: List[str] = MAIN_METRICS,
) -> None:
    """Generate exp_vs_vuln grids for each requested metric."""
    print("Generating Explanation vs Vulnerability charts...")
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric in metrics:
        # Skip metric if not present in either dataframe
        has_exp = (not explanation_df.empty) and (metric in explanation_df.columns)
        has_vuln = (not vulnerability_df.empty) and (metric in vulnerability_df.columns)
        if not (has_exp or has_vuln):
            print(f"  Skipping {metric}: metric not present in data")
            continue

        out_path = output_dir / f"{metric}_exp_vs_vuln_grid.png"
        plot_exp_vs_vuln_grid(metric, explanation_df, vulnerability_df, out_path, attack_name="pgd")

    print(f"✓ Exp_vs_vuln charts saved to: {output_dir}")


# =============================================================================
# Main Execution
# =============================================================================

def main() -> None:
    """Main entry point for the analysis runner."""
    global ANALYSIS_DIR, OUTPUT_DIR, TABLES_DIR, CHARTS_DIR, EXPLANATION_CHARTS_DIR, VULNERABILITY_CHARTS_DIR
    
    parser = argparse.ArgumentParser(
        description='Analyze metrics from visualize_vulnerability.py output',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default=str(DEFAULT_ANALYSIS_DIR),
        help=f'Input analysis directory (default: {DEFAULT_ANALYSIS_DIR})'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f'Output directory for results (default: {DEFAULT_OUTPUT_DIR})'
    )
    
    args = parser.parse_args()
    
    # Set directories from args
    ANALYSIS_DIR = Path(args.input)
    OUTPUT_DIR = Path(args.output)
    TABLES_DIR = OUTPUT_DIR / "tables"
    CHARTS_DIR = OUTPUT_DIR / "charts"
    EXPLANATION_CHARTS_DIR = CHARTS_DIR / "explanation"
    VULNERABILITY_CHARTS_DIR = CHARTS_DIR / "vulnerability"
    
    print("=" * 60)
    print("Forgery-Localization Pipeline Analysis Runner")
    print("=" * 60)
    print(f"Input directory: {ANALYSIS_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Step 1: Create output directories
    directories = [
        OUTPUT_DIR,
        TABLES_DIR,
        CHARTS_DIR,
        EXPLANATION_CHARTS_DIR,
        VULNERABILITY_CHARTS_DIR,
        CHARTS_DIR / "exp_vs_vuln",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directories created at: {OUTPUT_DIR}")
    print()
    
    # Step 2: Load all data
    explanation_df, vulnerability_df = load_all_data(ANALYSIS_DIR)
    print()
    
    if explanation_df.empty and vulnerability_df.empty:
        print("ERROR: No data found in the analysis directory.")
        print(f"Please ensure CSV files exist in: {ANALYSIS_DIR}")
        return
    
    # Step 3: Compute aggregations and save tables
    tables = generate_all_aggregations(explanation_df, vulnerability_df)
    save_tables(tables, TABLES_DIR)
    print()
    
    # Step 4: Compute shared axis ranges for paired plots
    shared_ranges = compute_shared_ranges(explanation_df, vulnerability_df)
    
    # Step 5: Generate charts with shared ranges
    if not explanation_df.empty:
        generate_explanation_charts(explanation_df, EXPLANATION_CHARTS_DIR, shared_ranges)
        print()
    
    if not vulnerability_df.empty:
        generate_vulnerability_charts(vulnerability_df, VULNERABILITY_CHARTS_DIR, shared_ranges)
        print()

    # Generate explanation vs vulnerability comparison charts
    generate_exp_vs_vuln_charts(explanation_df, vulnerability_df, CHARTS_DIR / "exp_vs_vuln")
    print()
    
    # Summary
    print("=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"Output location: {OUTPUT_DIR}")
    print()
    print("Generated files:")
    
    # List generated files
    for root, dirs, files in os.walk(OUTPUT_DIR):
        level = root.replace(str(OUTPUT_DIR), "").count(os.sep)
        indent = "  " * level
        folder_name = os.path.basename(root)
        print(f"{indent}📁 {folder_name}/")
        sub_indent = "  " * (level + 1)
        for file in files:
            print(f"{sub_indent}📄 {file}")


if __name__ == "__main__":
    main()
