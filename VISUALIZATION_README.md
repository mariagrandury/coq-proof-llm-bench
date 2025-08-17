# Visualization Script Documentation

This document describes the `visualize_results.py` script that creates comprehensive visualizations for analyzing Coq proof generation results.

## Overview

The visualization script creates 5 different plots that provide insights into model performance across different dimensions:

1. **Overall Performance Comparison** - Bar chart ranking all models by Pass@5 rate
2. **Learning Curve Analysis** - Line plots showing how models improve across attempts
3. **Language Performance Comparison** - Heatmap showing performance across languages
4. **Category Performance Comparison** - Grouped bar chart for different proof categories
5. **Top 5 Models Detailed Analysis** - Stratified breakdown for the best performing models

## Usage

### Option 1: Using the wrapper script (recommended)
```bash
./visualize_results.py
```

### Option 2: Using Python directly
```bash
python3.11 -m src.visualize_results
```

### Option 3: Using the module from any directory
```bash
python3.11 -m src.visualize_results
```

## Generated Plots

### 1. Overall Performance Comparison (`01_overall_performance.png`)
- **Purpose**: Compare all models' final performance (Pass@5 rate)
- **Type**: Bar chart
- **X-axis**: Models (sorted by performance)
- **Y-axis**: Pass@5 rate (0.0 to 1.0)
- **Insights**: 
  - Quick ranking of model performance
  - Performance gaps between models
  - Overall success rates

### 2. Learning Curves (`02_learning_curves.png`)
- **Purpose**: Show how models improve across multiple attempts
- **Type**: Line plot with markers
- **X-axis**: Attempt number (1st, 2nd, 3rd, 4th, 5th)
- **Y-axis**: Pass rate (0.0 to 1.0)
- **Insights**:
  - Learning efficiency of each model
  - Diminishing returns from additional attempts
  - Models that benefit most from multiple tries

### 3. Language Performance Comparison (`03_language_performance.png`)
- **Purpose**: Compare model performance across different languages
- **Type**: Heatmap
- **X-axis**: Languages (en, es, fr)
- **Y-axis**: Models
- **Color scale**: Pass@5 rate (red = low, green = high)
- **Insights**:
  - Language-specific strengths/weaknesses
  - Cross-lingual generalization
  - Performance consistency across languages

### 4. Category Performance Comparison (`04_category_performance.png`)
- **Purpose**: Compare performance on different proof categories
- **Type**: Grouped bar chart
- **X-axis**: Models
- **Y-axis**: Pass@5 rate
- **Groups**: Quantifiers vs Negation
- **Insights**:
  - Category-specific model strengths
  - Difficulty differences between categories
  - Model specialization patterns

### 5. Top 5 Models Detailed Analysis (`05_top5_models_analysis.png`)
- **Purpose**: Deep dive into the best performing models
- **Type**: 2x2 subplot grid
- **Subplots**:
  - **Top-left**: Learning curves for top 5 models
  - **Top-right**: Language performance breakdown
  - **Bottom-left**: Category performance breakdown
  - **Bottom-right**: Performance ranking visualization
- **Insights**:
  - Detailed comparison of top performers
  - Performance patterns across dimensions
  - Model strengths and weaknesses

## Data Structure

The script expects summary files in the following structure:
```
results/batch/
├── model1/
│   └── summary.json
├── model2/
│   └── summary.json
└── ...
```

Each `summary.json` should contain:
- `overall`: Overall performance metrics
- `by_lang`: Performance by language
- `by_category`: Performance by category
- `by_difficulty`: Performance by difficulty

## Dependencies

Required Python packages:
- `matplotlib` - Core plotting library
- `seaborn` - Statistical data visualization
- `pandas` - Data manipulation
- `numpy` - Numerical computing

Install with:
```bash
pip3 install matplotlib seaborn pandas numpy
```

## Output

All plots are saved in the `plots/` directory with descriptive filenames:
- `01_overall_performance.png`
- `02_learning_curves.png`
- `03_language_performance.png`
- `04_category_performance.png`
- `05_top5_models_analysis.png`
