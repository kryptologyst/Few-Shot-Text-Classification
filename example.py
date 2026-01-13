#!/usr/bin/env python3
"""
Example script demonstrating Few-Shot Text Classification

This script shows how to use the few-shot text classification system
with various examples and use cases.
"""

import logging
from pathlib import Path

from src.few_shot_classifier import FewShotTextClassifier, FewShotConfig
from src.dataset_generator import SyntheticDatasetGenerator
from src.config_manager import ConfigManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_zero_shot_classification():
    """Demonstrate zero-shot text classification."""
    print("\n" + "="*60)
    print("🔍 ZERO-SHOT TEXT CLASSIFICATION EXAMPLE")
    print("="*60)
    
    # Initialize classifier
    config = FewShotConfig(model_name="facebook/bart-large-mnli")
    classifier = FewShotTextClassifier(config)
    classifier.load_zero_shot_classifier()
    
    # Example texts
    texts = [
        "Apple announces new AI-powered health monitoring features for iPhone",
        "The Federal Reserve raises interest rates to combat inflation",
        "Scientists discover breakthrough treatment for Alzheimer's disease",
        "Manchester United wins the championship after thrilling final match",
        "New environmental regulations aim to reduce carbon emissions"
    ]
    
    # Candidate labels
    labels = ["technology", "healthcare", "finance", "politics", "sports", "environment"]
    
    print(f"📝 Classifying {len(texts)} texts with labels: {', '.join(labels)}")
    print()
    
    # Classify each text
    for i, text in enumerate(texts, 1):
        result = classifier.classify_text(text, labels)
        print(f"{i}. Text: {text}")
        print(f"   🎯 Predicted: {result.predicted_label}")
        print(f"   📊 Confidence: {result.confidence:.3f}")
        print(f"   📈 Top 3 scores:")
        sorted_scores = sorted(result.all_scores.items(), key=lambda x: x[1], reverse=True)
        for label, score in sorted_scores[:3]:
            print(f"      {label}: {score:.3f}")
        print()


def example_few_shot_training():
    """Demonstrate few-shot training."""
    print("\n" + "="*60)
    print("🎯 FEW-SHOT TRAINING EXAMPLE")
    print("="*60)
    
    # Generate synthetic dataset
    generator = SyntheticDatasetGenerator(seed=42)
    dataset = generator.generate_sentiment_dataset(100)
    
    # Split dataset
    train_data, val_data, test_data = generator.split_dataset(dataset, 0.6, 0.2, 0.2)
    
    print(f"📊 Generated sentiment dataset:")
    print(f"   Total samples: {len(dataset)}")
    print(f"   Training: {len(train_data)} samples")
    print(f"   Validation: {len(val_data)} samples")
    print(f"   Test: {len(test_data)} samples")
    
    # Get unique labels
    labels = list(set([ex['label'] for ex in dataset]))
    print(f"   Labels: {', '.join(labels)}")
    
    # Initialize classifier
    config = FewShotConfig(
        model_name="facebook/bart-large-mnli",
        num_epochs=2,
        batch_size=8
    )
    classifier = FewShotTextClassifier(config)
    
    # Prepare dataset
    train_dataset = classifier.prepare_few_shot_dataset(train_data, labels)
    val_dataset = classifier.prepare_few_shot_dataset(val_data, labels)
    
    print(f"\n🚀 Starting few-shot training with {len(train_data)} examples...")
    
    # Train model (commented out to avoid long training time in example)
    # classifier.fine_tune_model(train_dataset, val_dataset, output_dir="./models/example_model")
    
    print("✅ Training would complete here (commented out for demo)")
    print("   Model would be saved to ./models/example_model/")
    
    # Evaluate model
    test_dataset = classifier.prepare_few_shot_dataset(test_data, labels)
    print(f"\n📊 Evaluating on {len(test_data)} test examples...")
    
    # Evaluation would happen here
    print("✅ Evaluation would complete here (commented out for demo)")


def example_batch_classification():
    """Demonstrate batch classification."""
    print("\n" + "="*60)
    print("📦 BATCH CLASSIFICATION EXAMPLE")
    print("="*60)
    
    # Initialize classifier
    config = FewShotConfig()
    classifier = FewShotTextClassifier(config)
    classifier.load_zero_shot_classifier()
    
    # Batch of texts
    texts = [
        "The new smartphone has amazing camera features",
        "I love this product, it works perfectly",
        "This is terrible quality, very disappointed",
        "The service was okay, nothing special",
        "Outstanding customer support and fast delivery"
    ]
    
    labels = ["positive", "negative", "neutral"]
    
    print(f"📝 Batch classifying {len(texts)} texts for sentiment analysis")
    print(f"🏷️  Labels: {', '.join(labels)}")
    print()
    
    # Batch classify
    results = classifier.batch_classify(texts, labels)
    
    # Display results
    for i, result in enumerate(results, 1):
        print(f"{i}. Text: {result.text}")
        print(f"   🎯 Sentiment: {result.predicted_label}")
        print(f"   📊 Confidence: {result.confidence:.3f}")
        print()


def example_dataset_generation():
    """Demonstrate synthetic dataset generation."""
    print("\n" + "="*60)
    print("📊 SYNTHETIC DATASET GENERATION EXAMPLE")
    print("="*60)
    
    generator = SyntheticDatasetGenerator(seed=123)
    
    # Generate different types of datasets
    dataset_types = ["news", "sentiment", "topic"]
    
    for dataset_type in dataset_types:
        print(f"\n🔧 Generating {dataset_type} dataset...")
        
        if dataset_type == "news":
            dataset = generator.generate_news_dataset(50)
        elif dataset_type == "sentiment":
            dataset = generator.generate_sentiment_dataset(50)
        elif dataset_type == "topic":
            dataset = generator.generate_topic_dataset(50)
        
        # Analyze dataset
        labels = set([ex['label'] for ex in dataset])
        avg_length = sum(len(ex['text']) for ex in dataset) / len(dataset)
        
        print(f"   📊 Generated {len(dataset)} samples")
        print(f"   🏷️  Labels: {', '.join(sorted(labels))}")
        print(f"   📏 Average text length: {avg_length:.1f} characters")
        
        # Show sample
        print(f"   📝 Sample: {dataset[0]['text']}")
        print(f"   🏷️  Label: {dataset[0]['label']}")


def example_configuration_management():
    """Demonstrate configuration management."""
    print("\n" + "="*60)
    print("⚙️ CONFIGURATION MANAGEMENT EXAMPLE")
    print("="*60)
    
    # Initialize config manager
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    print("📋 Current configuration:")
    print(f"   Model: {config.model.name}")
    print(f"   Max length: {config.model.max_length}")
    print(f"   Batch size: {config.model.batch_size}")
    print(f"   Learning rate: {config.model.learning_rate}")
    print(f"   Dataset type: {config.data.dataset_type}")
    print(f"   Number of samples: {config.data.num_samples}")
    
    # Update configuration
    print("\n🔄 Updating configuration...")
    updates = {
        'model': {
            'name': 'microsoft/DialoGPT-medium',
            'max_length': 256,
            'batch_size': 32
        },
        'data': {
            'num_samples': 2000,
            'dataset_type': 'sentiment'
        }
    }
    
    config_manager.update_config(updates)
    updated_config = config_manager.get_config()
    
    print("📋 Updated configuration:")
    print(f"   Model: {updated_config.model.name}")
    print(f"   Max length: {updated_config.model.max_length}")
    print(f"   Batch size: {updated_config.model.batch_size}")
    print(f"   Dataset type: {updated_config.data.dataset_type}")
    print(f"   Number of samples: {updated_config.data.num_samples}")


def main():
    """Run all examples."""
    print("🤖 FEW-SHOT TEXT CLASSIFICATION - EXAMPLES")
    print("="*60)
    print("This script demonstrates various features of the few-shot")
    print("text classification system. Each example shows different")
    print("use cases and capabilities.")
    
    try:
        # Run examples
        example_zero_shot_classification()
        example_few_shot_training()
        example_batch_classification()
        example_dataset_generation()
        example_configuration_management()
        
        print("\n" + "="*60)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\n🚀 Next steps:")
        print("   1. Try the web interface: streamlit run web_app/app.py")
        print("   2. Use the CLI: python cli.py interactive")
        print("   3. Explore the source code in src/")
        print("   4. Run tests: python -m pytest tests/")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        print(f"\n❌ Example failed: {e}")
        print("Please check your installation and dependencies.")


if __name__ == "__main__":
    main()
