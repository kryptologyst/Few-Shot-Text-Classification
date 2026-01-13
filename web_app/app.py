"""
Streamlit Web Application for Few-Shot Text Classification

This module provides an interactive web interface for the few-shot text classification system.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any
import json
import logging
from pathlib import Path

from src.few_shot_classifier import FewShotTextClassifier, FewShotConfig, ClassificationResult
from src.dataset_generator import SyntheticDatasetGenerator
from src.config_manager import ConfigManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Few-Shot Text Classification",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'classifier' not in st.session_state:
        st.session_state.classifier = None
    if 'config_manager' not in st.session_state:
        st.session_state.config_manager = ConfigManager()
    if 'dataset_generator' not in st.session_state:
        st.session_state.dataset_generator = SyntheticDatasetGenerator()
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'dataset' not in st.session_state:
        st.session_state.dataset = None


def load_configuration():
    """Load and display configuration."""
    st.sidebar.header("⚙️ Configuration")
    
    config_manager = st.session_state.config_manager
    config = config_manager.get_config()
    
    # Model configuration
    st.sidebar.subheader("Model Settings")
    model_name = st.sidebar.selectbox(
        "Model",
        [
            "facebook/bart-large-mnli",
            "microsoft/DialoGPT-medium",
            "distilbert-base-uncased",
            "roberta-base"
        ],
        index=0
    )
    
    max_length = st.sidebar.slider("Max Length", 128, 1024, 512, 64)
    batch_size = st.sidebar.selectbox("Batch Size", [8, 16, 32], index=1)
    learning_rate = st.sidebar.slider("Learning Rate", 1e-6, 1e-3, 2e-5, 1e-6, format="%.0e")
    
    # Data configuration
    st.sidebar.subheader("Data Settings")
    dataset_type = st.sidebar.selectbox(
        "Dataset Type",
        ["news", "sentiment", "topic"],
        index=0
    )
    
    num_samples = st.sidebar.slider("Number of Samples", 100, 5000, 1000, 100)
    
    # Update configuration
    config_updates = {
        'model': {
            'name': model_name,
            'max_length': max_length,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        },
        'data': {
            'dataset_type': dataset_type,
            'num_samples': num_samples
        }
    }
    
    config_manager.update_config(config_updates)
    
    return config_manager.get_config()


def create_classifier(config):
    """Create and initialize the classifier."""
    try:
        few_shot_config = FewShotConfig(
            model_name=config.model.name,
            max_length=config.model.max_length,
            batch_size=config.model.batch_size,
            learning_rate=config.model.learning_rate,
            device=config.model.device
        )
        
        classifier = FewShotTextClassifier(few_shot_config)
        classifier.load_zero_shot_classifier()
        
        st.session_state.classifier = classifier
        return classifier
    except Exception as e:
        st.error(f"Failed to create classifier: {e}")
        return None


def generate_dataset(config):
    """Generate synthetic dataset."""
    generator = st.session_state.dataset_generator
    
    with st.spinner("Generating synthetic dataset..."):
        if config.data.dataset_type == "news":
            dataset = generator.generate_news_dataset(config.data.num_samples)
        elif config.data.dataset_type == "sentiment":
            dataset = generator.generate_sentiment_dataset(config.data.num_samples)
        elif config.data.dataset_type == "topic":
            dataset = generator.generate_topic_dataset(config.data.num_samples)
        else:
            st.error(f"Unknown dataset type: {config.data.dataset_type}")
            return None
    
    st.session_state.dataset = dataset
    return dataset


def display_dataset_info(dataset):
    """Display dataset information and statistics."""
    if not dataset:
        return
    
    st.subheader("📊 Dataset Information")
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(dataset)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Samples", len(dataset))
    
    with col2:
        st.metric("Unique Labels", df['label'].nunique())
    
    with col3:
        st.metric("Average Text Length", f"{df['text'].str.len().mean():.1f}")
    
    # Label distribution
    st.subheader("Label Distribution")
    label_counts = df['label'].value_counts()
    
    fig = px.pie(
        values=label_counts.values,
        names=label_counts.index,
        title="Distribution of Labels"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Text length distribution
    st.subheader("Text Length Distribution")
    df['text_length'] = df['text'].str.len()
    
    fig = px.histogram(
        df,
        x='text_length',
        title="Distribution of Text Lengths",
        nbins=20
    )
    fig.update_xaxis(title="Text Length (characters)")
    fig.update_yaxis(title="Count")
    st.plotly_chart(fig, use_container_width=True)
    
    # Sample data
    st.subheader("Sample Data")
    st.dataframe(df.head(10), use_container_width=True)


def classify_text_interface():
    """Interface for text classification."""
    st.header("🔍 Text Classification")
    
    classifier = st.session_state.classifier
    if not classifier:
        st.error("Please initialize the classifier first.")
        return
    
    # Input text
    text_input = st.text_area(
        "Enter text to classify:",
        value="The company just launched a new AI product for healthcare.",
        height=100
    )
    
    # Candidate labels
    st.subheader("Candidate Labels")
    
    # Predefined label sets
    label_sets = {
        "News Categories": ["technology", "healthcare", "finance", "politics", "sports"],
        "Sentiment": ["positive", "negative", "neutral"],
        "Topics": ["education", "entertainment", "travel", "food", "environment"],
        "Custom": []
    }
    
    selected_set = st.selectbox("Choose label set:", list(label_sets.keys()))
    
    if selected_set == "Custom":
        custom_labels = st.text_input(
            "Enter custom labels (comma-separated):",
            value="technology, healthcare, finance"
        )
        candidate_labels = [label.strip() for label in custom_labels.split(",") if label.strip()]
    else:
        candidate_labels = label_sets[selected_set]
        st.write(f"Labels: {', '.join(candidate_labels)}")
    
    # Classification options
    col1, col2 = st.columns(2)
    with col1:
        multi_label = st.checkbox("Allow multiple labels", value=False)
    with col2:
        batch_mode = st.checkbox("Batch mode", value=False)
    
    # Classify button
    if st.button("🚀 Classify Text", type="primary"):
        if not text_input.strip():
            st.error("Please enter some text to classify.")
            return
        
        if not candidate_labels:
            st.error("Please provide at least one candidate label.")
            return
        
        try:
            with st.spinner("Classifying text..."):
                if batch_mode:
                    # Split text by lines for batch processing
                    texts = [line.strip() for line in text_input.split('\n') if line.strip()]
                    results = classifier.batch_classify(texts, candidate_labels, multi_label)
                else:
                    result = classifier.classify_text(text_input, candidate_labels, multi_label)
                    results = [result]
            
            # Display results
            st.success(f"Classification completed! Processed {len(results)} text(s).")
            
            for i, result in enumerate(results):
                st.subheader(f"Result {i+1}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Label", result.predicted_label)
                with col2:
                    st.metric("Confidence", f"{result.confidence:.3f}")
                
                # Confidence scores for all labels
                st.subheader("Confidence Scores")
                scores_df = pd.DataFrame([
                    {"Label": label, "Score": score}
                    for label, score in result.all_scores.items()
                ]).sort_values("Score", ascending=False)
                
                fig = px.bar(
                    scores_df,
                    x="Score",
                    y="Label",
                    orientation="h",
                    title="Confidence Scores by Label"
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Store results
                st.session_state.results.extend(results)
        
        except Exception as e:
            st.error(f"Classification failed: {e}")


def few_shot_training_interface():
    """Interface for few-shot training."""
    st.header("🎯 Few-Shot Training")
    
    classifier = st.session_state.classifier
    dataset = st.session_state.dataset
    
    if not classifier:
        st.error("Please initialize the classifier first.")
        return
    
    if not dataset:
        st.error("Please generate a dataset first.")
        return
    
    st.subheader("Training Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        num_epochs = st.slider("Number of Epochs", 1, 10, 3)
        batch_size = st.selectbox("Training Batch Size", [8, 16, 32], index=1)
    
    with col2:
        learning_rate = st.slider("Learning Rate", 1e-6, 1e-3, 2e-5, 1e-6, format="%.0e")
        warmup_steps = st.slider("Warmup Steps", 0, 500, 100)
    
    # Sample selection for few-shot training
    st.subheader("Few-Shot Examples")
    
    # Get unique labels
    df = pd.DataFrame(dataset)
    unique_labels = df['label'].unique()
    
    st.write(f"Available labels: {', '.join(unique_labels)}")
    
    # Select examples per label
    examples_per_label = st.slider("Examples per label", 1, 20, 5)
    
    if st.button("🎯 Start Few-Shot Training", type="primary"):
        try:
            # Prepare few-shot examples
            few_shot_examples = []
            for label in unique_labels:
                label_examples = df[df['label'] == label].sample(
                    min(examples_per_label, len(df[df['label'] == label])),
                    random_state=42
                )
                few_shot_examples.extend(label_examples.to_dict('records'))
            
            st.info(f"Using {len(few_shot_examples)} examples for few-shot training")
            
            # Prepare dataset
            train_dataset = classifier.prepare_few_shot_dataset(
                few_shot_examples, 
                unique_labels.tolist()
            )
            
            # Split dataset
            train_size = int(len(train_dataset) * 0.8)
            train_data = train_dataset.select(range(train_size))
            eval_data = train_dataset.select(range(train_size, len(train_dataset)))
            
            # Update classifier config
            classifier.config.num_epochs = num_epochs
            classifier.config.batch_size = batch_size
            classifier.config.learning_rate = learning_rate
            classifier.config.warmup_steps = warmup_steps
            
            # Train the model
            with st.spinner("Training model..."):
                classifier.fine_tune_model(
                    train_data,
                    eval_data,
                    output_dir="./models/few_shot_model"
                )
            
            st.success("Few-shot training completed successfully!")
            
            # Evaluate the model
            if len(eval_data) > 0:
                metrics = classifier.evaluate_model(eval_data)
                
                st.subheader("Evaluation Results")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                with col2:
                    st.metric("Precision", f"{metrics['precision']:.3f}")
                with col3:
                    st.metric("Recall", f"{metrics['recall']:.3f}")
                with col4:
                    st.metric("F1 Score", f"{metrics['f1']:.3f}")
        
        except Exception as e:
            st.error(f"Training failed: {e}")


def results_analysis():
    """Display and analyze classification results."""
    st.header("📈 Results Analysis")
    
    results = st.session_state.results
    if not results:
        st.info("No classification results available yet.")
        return
    
    st.subheader(f"Total Classifications: {len(results)}")
    
    # Convert results to DataFrame
    results_data = []
    for result in results:
        results_data.append({
            'Text': result.text[:100] + "..." if len(result.text) > 100 else result.text,
            'Predicted Label': result.predicted_label,
            'Confidence': result.confidence,
            'Model': result.model_name
        })
    
    df_results = pd.DataFrame(results_data)
    
    # Display results table
    st.subheader("Classification Results")
    st.dataframe(df_results, use_container_width=True)
    
    # Analysis charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Label distribution
        label_counts = df_results['Predicted Label'].value_counts()
        fig = px.pie(
            values=label_counts.values,
            names=label_counts.index,
            title="Predicted Label Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Confidence distribution
        fig = px.histogram(
            df_results,
            x='Confidence',
            title="Confidence Score Distribution",
            nbins=20
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Download results
    st.subheader("Export Results")
    
    csv_data = df_results.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv_data,
        file_name="classification_results.csv",
        mime="text/csv"
    )


def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Few-Shot Text Classification</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Interactive demo for few-shot learning using state-of-the-art transformer models
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load configuration
    config = load_configuration()
    
    # Initialize classifier
    if st.sidebar.button("🚀 Initialize Classifier", type="primary"):
        classifier = create_classifier(config)
        if classifier:
            st.sidebar.success("Classifier initialized successfully!")
    
    # Generate dataset
    if st.sidebar.button("📊 Generate Dataset"):
        dataset = generate_dataset(config)
        if dataset:
            st.sidebar.success(f"Generated {len(dataset)} samples!")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Classify", "📊 Dataset", "🎯 Training", "📈 Results"])
    
    with tab1:
        classify_text_interface()
    
    with tab2:
        display_dataset_info(st.session_state.dataset)
    
    with tab3:
        few_shot_training_interface()
    
    with tab4:
        results_analysis()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>Built with ❤️ using Streamlit, Transformers, and Hugging Face</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
