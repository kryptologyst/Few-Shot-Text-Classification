#!/usr/bin/env python3
"""
Command Line Interface for Few-Shot Text Classification

This module provides a command-line interface for the few-shot text classification system.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import logging

from src.few_shot_classifier import FewShotTextClassifier, FewShotConfig
from src.dataset_generator import SyntheticDatasetGenerator
from src.config_manager import ConfigManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def classify_text_cli(args):
    """Classify text using command line interface."""
    try:
        # Create classifier
        config = FewShotConfig(
            model_name=args.model,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=args.device
        )
        
        classifier = FewShotTextClassifier(config)
        classifier.load_zero_shot_classifier()
        
        # Read text
        if args.text:
            text = args.text
        elif args.file:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        else:
            print("Please provide either --text or --file argument")
            return
        
        # Parse labels
        labels = [label.strip() for label in args.labels.split(',')]
        
        # Classify
        result = classifier.classify_text(text, labels, args.multi_label)
        
        # Display results
        print(f"\n📝 Text: {text}")
        print(f"🏷️  Predicted Label: {result.predicted_label}")
        print(f"🎯 Confidence: {result.confidence:.3f}")
        print(f"\n📊 All Scores:")
        for label, score in result.all_scores.items():
            print(f"  {label}: {score:.3f}")
        
        # Save results if requested
        if args.output:
            results = [result]
            classifier.save_results(results, args.output)
            print(f"\n💾 Results saved to {args.output}")
    
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        sys.exit(1)


def generate_dataset_cli(args):
    """Generate synthetic dataset using command line interface."""
    try:
        generator = SyntheticDatasetGenerator(seed=args.seed)
        
        # Generate dataset
        if args.type == "news":
            dataset = generator.generate_news_dataset(args.samples)
        elif args.type == "sentiment":
            dataset = generator.generate_sentiment_dataset(args.samples)
        elif args.type == "topic":
            dataset = generator.generate_topic_dataset(args.samples)
        else:
            print(f"Unknown dataset type: {args.type}")
            return
        
        # Split dataset
        train_data, val_data, test_data = generator.split_dataset(
            dataset, args.train_ratio, args.val_ratio, args.test_ratio
        )
        
        # Save datasets
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generator.save_dataset(train_data, output_dir / "train.json")
        generator.save_dataset(val_data, output_dir / "val.json")
        generator.save_dataset(test_data, output_dir / "test.json")
        
        print(f"\n✅ Generated {args.type} dataset:")
        print(f"  📊 Total samples: {len(dataset)}")
        print(f"  🚂 Training: {len(train_data)} samples")
        print(f"  ✅ Validation: {len(val_data)} samples")
        print(f"  🧪 Test: {len(test_data)} samples")
        print(f"  💾 Saved to: {output_dir}")
    
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        sys.exit(1)


def train_model_cli(args):
    """Train model using command line interface."""
    try:
        # Load dataset
        generator = SyntheticDatasetGenerator()
        train_data = generator.load_dataset(args.train_file)
        val_data = generator.load_dataset(args.val_file) if args.val_file else None
        
        # Get unique labels
        labels = list(set([ex['label'] for ex in train_data]))
        
        # Create classifier
        config = FewShotConfig(
            model_name=args.model,
            max_length=args.max_length,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.epochs,
            device=args.device
        )
        
        classifier = FewShotTextClassifier(config)
        
        # Prepare dataset
        train_dataset = classifier.prepare_few_shot_dataset(train_data, labels)
        val_dataset = classifier.prepare_few_shot_dataset(val_data, labels) if val_data else None
        
        # Train model
        print(f"\n🚀 Starting training with {len(train_data)} examples...")
        classifier.fine_tune_model(
            train_dataset,
            val_dataset,
            output_dir=args.output_dir
        )
        
        print(f"\n✅ Training completed! Model saved to {args.output_dir}")
        
        # Evaluate if test data provided
        if args.test_file:
            test_data = generator.load_dataset(args.test_file)
            test_dataset = classifier.prepare_few_shot_dataset(test_data, labels)
            
            metrics = classifier.evaluate_model(test_dataset, args.output_dir)
            
            print(f"\n📊 Test Results:")
            print(f"  Accuracy: {metrics['accuracy']:.3f}")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")
            print(f"  F1 Score: {metrics['f1']:.3f}")
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


def interactive_mode():
    """Run interactive mode for text classification."""
    print("\n🤖 Few-Shot Text Classification - Interactive Mode")
    print("=" * 50)
    
    # Initialize classifier
    config = FewShotConfig()
    classifier = FewShotTextClassifier(config)
    classifier.load_zero_shot_classifier()
    
    print("✅ Classifier initialized!")
    
    while True:
        print("\n" + "=" * 50)
        text = input("📝 Enter text to classify (or 'quit' to exit): ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not text:
            print("❌ Please enter some text.")
            continue
        
        labels_input = input("🏷️  Enter labels (comma-separated): ").strip()
        if not labels_input:
            print("❌ Please enter at least one label.")
            continue
        
        labels = [label.strip() for label in labels_input.split(',')]
        
        try:
            result = classifier.classify_text(text, labels)
            
            print(f"\n🎯 Result:")
            print(f"  Predicted: {result.predicted_label}")
            print(f"  Confidence: {result.confidence:.3f}")
            print(f"\n📊 All Scores:")
            for label, score in result.all_scores.items():
                print(f"  {label}: {score:.3f}")
        
        except Exception as e:
            print(f"❌ Classification failed: {e}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Few-Shot Text Classification CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classify text with labels
  python cli.py classify --text "AI is revolutionizing healthcare" --labels "technology,healthcare,finance"
  
  # Classify text from file
  python cli.py classify --file input.txt --labels "positive,negative,neutral"
  
  # Generate synthetic dataset
  python cli.py generate --type news --samples 1000 --output-dir data/
  
  # Train model
  python cli.py train --train-file data/train.json --val-file data/val.json --epochs 3
  
  # Interactive mode
  python cli.py interactive
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Classify command
    classify_parser = subparsers.add_parser('classify', help='Classify text')
    classify_parser.add_argument('--text', help='Text to classify')
    classify_parser.add_argument('--file', help='File containing text to classify')
    classify_parser.add_argument('--labels', required=True, help='Comma-separated labels')
    classify_parser.add_argument('--model', default='facebook/bart-large-mnli', help='Model name')
    classify_parser.add_argument('--max-length', type=int, default=512, help='Max sequence length')
    classify_parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    classify_parser.add_argument('--device', default='auto', help='Device to use')
    classify_parser.add_argument('--multi-label', action='store_true', help='Allow multiple labels')
    classify_parser.add_argument('--output', help='Output file for results')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate synthetic dataset')
    generate_parser.add_argument('--type', choices=['news', 'sentiment', 'topic'], 
                                default='news', help='Dataset type')
    generate_parser.add_argument('--samples', type=int, default=1000, help='Number of samples')
    generate_parser.add_argument('--output-dir', default='data/', help='Output directory')
    generate_parser.add_argument('--train-ratio', type=float, default=0.7, help='Training ratio')
    generate_parser.add_argument('--val-ratio', type=float, default=0.15, help='Validation ratio')
    generate_parser.add_argument('--test-ratio', type=float, default=0.15, help='Test ratio')
    generate_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--train-file', required=True, help='Training data file')
    train_parser.add_argument('--val-file', help='Validation data file')
    train_parser.add_argument('--test-file', help='Test data file')
    train_parser.add_argument('--model', default='facebook/bart-large-mnli', help='Model name')
    train_parser.add_argument('--max-length', type=int, default=512, help='Max sequence length')
    train_parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    train_parser.add_argument('--learning-rate', type=float, default=2e-5, help='Learning rate')
    train_parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    train_parser.add_argument('--device', default='auto', help='Device to use')
    train_parser.add_argument('--output-dir', default='models/', help='Output directory')
    
    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Run interactive mode')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'classify':
        classify_text_cli(args)
    elif args.command == 'generate':
        generate_dataset_cli(args)
    elif args.command == 'train':
        train_model_cli(args)
    elif args.command == 'interactive':
        interactive_mode()


if __name__ == "__main__":
    main()
