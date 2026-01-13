"""
Utility functions for Few-Shot Text Classification

This module contains utility functions for data processing, visualization, and analysis.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


def create_confusion_matrix_plot(
    y_true: List[str], 
    y_pred: List[str], 
    labels: List[str],
    title: str = "Confusion Matrix"
) -> go.Figure:
    """
    Create an interactive confusion matrix plot.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of unique labels
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 12},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        width=600,
        height=500
    )
    
    return fig


def create_classification_report_plot(
    y_true: List[str], 
    y_pred: List[str], 
    labels: List[str]
) -> go.Figure:
    """
    Create an interactive classification report plot.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of unique labels
        
    Returns:
        Plotly figure object
    """
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True)
    
    # Extract metrics for each class
    classes = [label for label in labels if label in report]
    precision = [report[label]['precision'] for label in classes]
    recall = [report[label]['recall'] for label in classes]
    f1_score = [report[label]['f1-score'] for label in classes]
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Precision', 'Recall', 'F1-Score'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    fig.add_trace(
        go.Bar(x=classes, y=precision, name='Precision', marker_color='lightblue'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=classes, y=recall, name='Recall', marker_color='lightgreen'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(x=classes, y=f1_score, name='F1-Score', marker_color='lightcoral'),
        row=1, col=3
    )
    
    fig.update_layout(
        title="Classification Report",
        showlegend=False,
        height=400,
        width=900
    )
    
    return fig


def create_confidence_distribution_plot(
    confidences: List[float],
    labels: List[str],
    title: str = "Confidence Score Distribution"
) -> go.Figure:
    """
    Create a confidence score distribution plot.
    
    Args:
        confidences: List of confidence scores
        labels: List of corresponding labels
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    df = pd.DataFrame({
        'confidence': confidences,
        'label': labels
    })
    
    fig = px.box(
        df,
        x='label',
        y='confidence',
        title=title,
        color='label'
    )
    
    fig.update_layout(
        xaxis_title="Label",
        yaxis_title="Confidence Score",
        height=500,
        width=800
    )
    
    return fig


def create_text_length_analysis_plot(
    texts: List[str],
    labels: List[str],
    title: str = "Text Length Analysis"
) -> go.Figure:
    """
    Create text length analysis plots.
    
    Args:
        texts: List of texts
        labels: List of corresponding labels
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    df = pd.DataFrame({
        'text': texts,
        'label': labels,
        'length': [len(text) for text in texts]
    })
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Text Length Distribution', 'Average Length by Label'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Length distribution
    fig.add_trace(
        go.Histogram(x=df['length'], name='Length Distribution', nbinsx=20),
        row=1, col=1
    )
    
    # Average length by label
    avg_lengths = df.groupby('label')['length'].mean().reset_index()
    fig.add_trace(
        go.Bar(x=avg_lengths['label'], y=avg_lengths['length'], name='Avg Length'),
        row=1, col=2
    )
    
    fig.update_layout(
        title=title,
        height=400,
        width=1000,
        showlegend=False
    )
    
    return fig


def analyze_classification_results(
    results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Analyze classification results and return statistics.
    
    Args:
        results: List of classification results
        
    Returns:
        Dictionary with analysis statistics
    """
    if not results:
        return {}
    
    df = pd.DataFrame(results)
    
    analysis = {
        'total_classifications': len(results),
        'unique_labels': df['predicted_label'].nunique(),
        'average_confidence': df['confidence'].mean(),
        'confidence_std': df['confidence'].std(),
        'min_confidence': df['confidence'].min(),
        'max_confidence': df['confidence'].max(),
        'label_distribution': df['predicted_label'].value_counts().to_dict(),
        'confidence_by_label': df.groupby('predicted_label')['confidence'].agg(['mean', 'std', 'count']).to_dict()
    }
    
    return analysis


def export_results_to_excel(
    results: List[Dict[str, Any]], 
    filepath: str,
    include_analysis: bool = True
) -> None:
    """
    Export classification results to Excel file.
    
    Args:
        results: List of classification results
        filepath: Path to save Excel file
        include_analysis: Whether to include analysis sheet
    """
    df = pd.DataFrame(results)
    
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        # Main results sheet
        df.to_excel(writer, sheet_name='Results', index=False)
        
        if include_analysis:
            # Analysis sheet
            analysis = analyze_classification_results(results)
            
            analysis_data = []
            for key, value in analysis.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        analysis_data.append({
                            'Metric': f"{key}_{sub_key}",
                            'Value': sub_value
                        })
                else:
                    analysis_data.append({
                        'Metric': key,
                        'Value': value
                    })
            
            analysis_df = pd.DataFrame(analysis_data)
            analysis_df.to_excel(writer, sheet_name='Analysis', index=False)
    
    logger.info(f"Results exported to {filepath}")


def create_model_comparison_plot(
    model_results: Dict[str, List[Dict[str, Any]]],
    metric: str = 'accuracy'
) -> go.Figure:
    """
    Create a model comparison plot.
    
    Args:
        model_results: Dictionary mapping model names to results
        metric: Metric to compare
        
    Returns:
        Plotly figure object
    """
    model_names = []
    metric_values = []
    
    for model_name, results in model_results.items():
        if results:
            analysis = analyze_classification_results(results)
            if metric in analysis:
                model_names.append(model_name)
                metric_values.append(analysis[metric])
    
    fig = go.Figure(data=[
        go.Bar(x=model_names, y=metric_values, marker_color='lightblue')
    ])
    
    fig.update_layout(
        title=f"Model Comparison - {metric.title()}",
        xaxis_title="Model",
        yaxis_title=metric.title(),
        height=400,
        width=600
    )
    
    return fig


def validate_dataset_format(dataset: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """
    Validate dataset format and return validation results.
    
    Args:
        dataset: Dataset to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    if not dataset:
        errors.append("Dataset is empty")
        return False, errors
    
    # Check required keys
    required_keys = {'text', 'label'}
    for i, example in enumerate(dataset):
        if not isinstance(example, dict):
            errors.append(f"Example {i} is not a dictionary")
            continue
        
        missing_keys = required_keys - set(example.keys())
        if missing_keys:
            errors.append(f"Example {i} missing keys: {missing_keys}")
        
        # Check data types
        if 'text' in example and not isinstance(example['text'], str):
            errors.append(f"Example {i} 'text' is not a string")
        
        if 'label' in example and not isinstance(example['label'], str):
            errors.append(f"Example {i} 'label' is not a string")
    
    return len(errors) == 0, errors


def create_dataset_statistics_plot(dataset: List[Dict[str, Any]]) -> go.Figure:
    """
    Create dataset statistics visualization.
    
    Args:
        dataset: Dataset to analyze
        
    Returns:
        Plotly figure object
    """
    df = pd.DataFrame(dataset)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Label Distribution', 'Text Length Distribution', 
                       'Average Length by Label', 'Text Length vs Label'),
        specs=[[{"type": "pie"}, {"type": "histogram"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # Label distribution
    label_counts = df['label'].value_counts()
    fig.add_trace(
        go.Pie(labels=label_counts.index, values=label_counts.values, name="Labels"),
        row=1, col=1
    )
    
    # Text length distribution
    df['text_length'] = df['text'].str.len()
    fig.add_trace(
        go.Histogram(x=df['text_length'], name="Length Distribution"),
        row=1, col=2
    )
    
    # Average length by label
    avg_lengths = df.groupby('label')['text_length'].mean()
    fig.add_trace(
        go.Bar(x=avg_lengths.index, y=avg_lengths.values, name="Avg Length"),
        row=2, col=1
    )
    
    # Scatter plot
    fig.add_trace(
        go.Scatter(x=df['text_length'], y=df['label'], mode='markers', name="Length vs Label"),
        row=2, col=2
    )
    
    fig.update_layout(
        title="Dataset Statistics",
        height=800,
        width=1000,
        showlegend=False
    )
    
    return fig
