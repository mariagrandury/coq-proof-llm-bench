#!/usr/bin/env python3
"""
Visualization script for Coq proof generation results.
Creates comprehensive plots to analyze and compare model performance.
"""

import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style for better-looking plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def load_all_summaries(results_dir: str = "results/batch") -> Dict[str, Dict]:
    """Load all summary.json files from the results directory."""
    summaries = {}

    # Find all summary.json files
    summary_files = glob.glob(os.path.join(results_dir, "*/summary.json"))

    for file_path in summary_files:
        # Extract model name from path
        model_name = os.path.basename(os.path.dirname(file_path))

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
                summaries[model_name] = summary
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")

    return summaries


def create_overall_performance_plot(
    summaries: Dict[str, Dict], output_dir: str = "plots"
):
    """Plot 1: Overall performance comparison - Bar chart of pass5 rates."""
    plt.figure(figsize=(14, 8))

    # Extract pass5 rates and sort by performance
    model_performance = []
    for model, summary in summaries.items():
        pass5_rate = summary.get("overall", {}).get("pass5", 0.0)
        model_performance.append((model, pass5_rate))

    # Sort by performance (descending)
    model_performance.sort(key=lambda x: x[1], reverse=True)

    models, rates = zip(*model_performance)

    # Create bar plot
    bars = plt.bar(
        range(len(models)), rates, color="skyblue", edgecolor="navy", alpha=0.7
    )

    # Customize the plot
    plt.xlabel("Models", fontsize=12, fontweight="bold")
    plt.ylabel("Pass@5 Rate", fontsize=12, fontweight="bold")
    plt.title(
        "Overall Model Performance Comparison (Pass@5)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Rotate x-axis labels for better readability
    plt.xticks(range(len(models)), models, rotation=45, ha="right")

    # Add value labels on bars
    for i, (bar, rate) in enumerate(zip(bars, rates)):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{rate:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Add grid for better readability
    plt.grid(True, alpha=0.3, axis="y")
    plt.ylim(0, max(rates) * 1.1)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "01_overall_performance.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return model_performance


def create_learning_curve_plot(summaries: Dict[str, Dict], output_dir: str = "plots"):
    """Plot 2: Learning curve analysis - Line plot of pass rates across attempts."""
    plt.figure(figsize=(14, 8))

    # Define attempt labels
    attempts = ["pass1", "pass2", "pass3", "pass4", "pass5"]
    attempt_labels = ["1st", "2nd", "3rd", "4th", "5th"]

    # Plot each model's learning curve
    for model, summary in summaries.items():
        overall = summary.get("overall", {})
        rates = [overall.get(attempt, 0.0) for attempt in attempts]

        plt.plot(
            attempt_labels, rates, marker="o", linewidth=2, markersize=6, label=model
        )

    plt.xlabel("Attempt Number", fontsize=12, fontweight="bold")
    plt.ylabel("Pass Rate", fontsize=12, fontweight="bold")
    plt.title(
        "Learning Curves: Pass Rates Across Attempts",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "02_learning_curves.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def create_language_performance_plot(
    summaries: Dict[str, Dict], output_dir: str = "plots"
):
    """Plot 3: Language performance comparison - Heatmap across languages."""
    plt.figure(figsize=(16, 10))

    # Get all languages from the data
    languages = set()
    for summary in summaries.values():
        for lang in summary.get("by_lang", {}).keys():
            languages.add(lang)

    languages = sorted(list(languages))

    # Create data matrix for heatmap
    data_matrix = []
    model_names = []

    for model, summary in summaries.items():
        model_names.append(model)
        row = []
        for lang in languages:
            lang_data = summary.get("by_lang", {}).get(lang, {})
            pass5_rate = lang_data.get("pass5", 0.0)
            row.append(pass5_rate)
        data_matrix.append(row)

    # Create heatmap
    data_array = np.array(data_matrix)

    # Create custom colormap
    cmap = sns.color_palette("RdYlGn", as_cmap=True)

    sns.heatmap(
        data_array,
        xticklabels=languages,
        yticklabels=model_names,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        cbar_kws={"label": "Pass@5 Rate"},
        linewidths=0.5,
    )

    plt.xlabel("Languages", fontsize=12, fontweight="bold")
    plt.ylabel("Models", fontsize=12, fontweight="bold")
    plt.title(
        "Language Performance Comparison (Pass@5)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "03_language_performance.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def create_category_performance_plot(
    summaries: Dict[str, Dict], output_dir: str = "plots"
):
    """Plot 4: Category performance comparison - Grouped bar chart for quantifiers vs negation."""
    plt.figure(figsize=(16, 10))

    # Extract data for categories
    categories = ["quantifiers", "negation"]
    models = list(summaries.keys())

    # Prepare data
    quantifier_data = []
    negation_data = []

    for model in models:
        summary = summaries[model]
        by_cat = summary.get("by_category", {})

        quantifier_rate = by_cat.get("quantifiers", {}).get("pass5", 0.0)
        negation_rate = by_cat.get("negation", {}).get("pass5", 0.0)

        quantifier_data.append(quantifier_rate)
        negation_data.append(negation_rate)

    # Set up the plot
    x = np.arange(len(models))
    width = 0.35

    # Create grouped bars
    bars1 = plt.bar(
        x - width / 2,
        quantifier_data,
        width,
        label="Quantifiers",
        color="lightcoral",
        alpha=0.8,
        edgecolor="darkred",
    )
    bars2 = plt.bar(
        x + width / 2,
        negation_data,
        width,
        label="Negation",
        color="lightblue",
        alpha=0.8,
        edgecolor="darkblue",
    )

    # Customize the plot
    plt.xlabel("Models", fontsize=12, fontweight="bold")
    plt.ylabel("Pass@5 Rate", fontsize=12, fontweight="bold")
    plt.title(
        "Category Performance Comparison (Pass@5)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    plt.xticks(x, models, rotation=45, ha="right")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.ylim(0, max(max(quantifier_data), max(negation_data)) * 1.1)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "04_category_performance.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def create_top_models_stratified_plot(
    summaries: Dict[str, Dict],
    top_models: List[Tuple[str, float]],
    output_dir: str = "plots",
):
    """Plot 5: Top 5 models detailed analysis - Stratified breakdown."""
    if len(top_models) < 5:
        print(f"Warning: Only {len(top_models)} models available for top 5 analysis")
        return

    # Take top 5 models
    top_5_models = top_models[:5]
    model_names = [model[0] for model in top_5_models]

    # Create subplots for different stratifications
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(
        "Top 5 Models: Detailed Performance Analysis", fontsize=20, fontweight="bold"
    )

    # Subplot 1: Overall performance across attempts
    ax1 = axes[0, 0]
    attempts = ["pass1", "pass2", "pass3", "pass4", "pass5"]
    attempt_labels = ["1st", "2nd", "3rd", "4th", "5th"]

    for model_name in model_names:
        summary = summaries[model_name]
        overall = summary.get("overall", {})
        rates = [overall.get(attempt, 0.0) for attempt in attempts]
        ax1.plot(
            attempt_labels,
            rates,
            marker="o",
            linewidth=2,
            markersize=6,
            label=model_name,
        )

    ax1.set_xlabel("Attempt Number", fontweight="bold")
    ax1.set_ylabel("Pass Rate", fontweight="bold")
    ax1.set_title("Learning Curves (Top 5 Models)", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.0)

    # Subplot 2: Language performance
    ax2 = axes[0, 1]
    languages = ["en", "es", "fr"]
    lang_labels = ["English", "Spanish", "French"]

    x = np.arange(len(languages))
    width = 0.15

    for i, model_name in enumerate(model_names):
        summary = summaries[model_name]
        by_lang = summary.get("by_lang", {})
        rates = [by_lang.get(lang, {}).get("pass5", 0.0) for lang in languages]
        ax2.bar(x + i * width, rates, width, label=model_name, alpha=0.8)

    ax2.set_xlabel("Languages", fontweight="bold")
    ax2.set_ylabel("Pass@5 Rate", fontweight="bold")
    ax2.set_title("Language Performance (Top 5 Models)", fontweight="bold")
    ax2.set_xticks(x + width * 2)
    ax2.set_xticklabels(lang_labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    # Subplot 3: Category performance
    ax3 = axes[1, 0]
    categories = ["quantifiers", "negation"]
    cat_labels = ["Quantifiers", "Negation"]

    x = np.arange(len(categories))

    for i, model_name in enumerate(model_names):
        summary = summaries[model_name]
        by_cat = summary.get("by_category", {})
        rates = [by_cat.get(cat, {}).get("pass5", 0.0) for cat in categories]
        ax3.bar(x + i * width, rates, width, label=model_name, alpha=0.8)

    ax3.set_xlabel("Categories", fontweight="bold")
    ax3.set_ylabel("Pass@5 Rate", fontweight="bold")
    ax3.set_title("Category Performance (Top 5 Models)", fontweight="bold")
    ax3.set_xticks(x + width * 2)
    ax3.set_xticklabels(cat_labels)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")

    # Subplot 4: Performance ranking
    ax4 = axes[1, 1]
    performance_values = [model[1] for model in top_5_models]
    model_names_short = [
        name.split("_")[0] if "_" in name else name[:10] for name in model_names
    ]

    bars = ax4.barh(
        range(len(model_names_short)),
        performance_values,
        color="lightgreen",
        edgecolor="darkgreen",
        alpha=0.8,
    )

    ax4.set_yticks(range(len(model_names_short)))
    ax4.set_yticklabels(model_names_short)
    ax4.set_xlabel("Pass@5 Rate", fontweight="bold")
    ax4.set_title("Performance Ranking (Top 5 Models)", fontweight="bold")
    ax4.grid(True, alpha=0.3, axis="x")

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, performance_values)):
        ax4.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            ha="left",
            va="center",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "05_top5_models_analysis.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def main():
    """Main function to create all visualizations."""
    print("Loading model summaries...")
    summaries = load_all_summaries()

    if not summaries:
        print("Error: No summaries found!")
        return

    print(f"Loaded {len(summaries)} model summaries")

    # Create output directory
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)

    print("Creating visualizations...")

    # Plot 1: Overall performance comparison
    print("1. Creating overall performance plot...")
    top_models = create_overall_performance_plot(summaries, output_dir)

    # Plot 2: Learning curve analysis
    print("2. Creating learning curve plot...")
    create_learning_curve_plot(summaries, output_dir)

    # Plot 3: Language performance comparison
    print("3. Creating language performance plot...")
    create_language_performance_plot(summaries, output_dir)

    # Plot 4: Category performance comparison
    print("4. Creating category performance plot...")
    create_category_performance_plot(summaries, output_dir)

    # Plot 5: Top 5 models detailed analysis
    print("5. Creating top 5 models stratified plot...")
    create_top_models_stratified_plot(summaries, top_models, output_dir)

    print(f"\nAll visualizations saved to '{output_dir}/' directory!")
    print("\nGenerated plots:")
    print("1. 01_overall_performance.png - Overall model performance comparison")
    print("2. 02_learning_curves.png - Learning curves across attempts")
    print("3. 03_language_performance.png - Language performance heatmap")
    print("4. 04_category_performance.png - Category performance comparison")
    print("5. 05_top5_models_analysis.png - Top 5 models detailed analysis")


if __name__ == "__main__":
    main()
