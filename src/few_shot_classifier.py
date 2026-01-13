"""
Few-Shot Text Classification Module

This module provides a comprehensive implementation of few-shot learning for text classification
using various transformer models and techniques.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Data class for storing classification results."""
    text: str
    predicted_label: str
    confidence: float
    all_scores: Dict[str, float]
    model_name: str


@dataclass
class FewShotConfig:
    """Configuration class for few-shot learning parameters."""
    model_name: str = "facebook/bart-large-mnli"
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    device: str = "auto"


class FewShotTextClassifier:
    """
    A comprehensive few-shot text classifier supporting multiple approaches:
    - Zero-shot classification
    - Few-shot fine-tuning
    - Prompt-based learning
    """
    
    def __init__(self, config: Optional[FewShotConfig] = None):
        """
        Initialize the few-shot text classifier.
        
        Args:
            config: Configuration object with model parameters
        """
        self.config = config or FewShotConfig()
        self.device = self._get_device()
        self.classifier = None
        self.tokenizer = None
        self.model = None
        self.training_history = []
        
        logger.info(f"Initialized FewShotTextClassifier with device: {self.device}")
    
    def _get_device(self) -> str:
        """Determine the best available device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return self.config.device
    
    def load_zero_shot_classifier(self) -> None:
        """Load a zero-shot classification pipeline."""
        try:
            self.classifier = pipeline(
                "zero-shot-classification",
                model=self.config.model_name,
                device=0 if self.device == "cuda" else -1
            )
            logger.info(f"Loaded zero-shot classifier: {self.config.model_name}")
        except Exception as e:
            logger.error(f"Failed to load zero-shot classifier: {e}")
            raise
    
    def classify_text(
        self, 
        text: str, 
        candidate_labels: List[str],
        multi_label: bool = False
    ) -> ClassificationResult:
        """
        Classify text using zero-shot classification.
        
        Args:
            text: Input text to classify
            candidate_labels: List of possible labels
            multi_label: Whether to allow multiple labels
            
        Returns:
            ClassificationResult object with predictions
        """
        if self.classifier is None:
            self.load_zero_shot_classifier()
        
        try:
            result = self.classifier(
                text, 
                candidate_labels, 
                multi_label=multi_label
            )
            
            return ClassificationResult(
                text=text,
                predicted_label=result['labels'][0],
                confidence=result['scores'][0],
                all_scores=dict(zip(result['labels'], result['scores'])),
                model_name=self.config.model_name
            )
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            raise
    
    def batch_classify(
        self, 
        texts: List[str], 
        candidate_labels: List[str],
        multi_label: bool = False
    ) -> List[ClassificationResult]:
        """
        Classify multiple texts in batch.
        
        Args:
            texts: List of texts to classify
            candidate_labels: List of possible labels
            multi_label: Whether to allow multiple labels
            
        Returns:
            List of ClassificationResult objects
        """
        results = []
        for text in texts:
            try:
                result = self.classify_text(text, candidate_labels, multi_label)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to classify text '{text[:50]}...': {e}")
                continue
        
        logger.info(f"Successfully classified {len(results)}/{len(texts)} texts")
        return results
    
    def prepare_few_shot_dataset(
        self, 
        examples: List[Dict[str, Any]], 
        candidate_labels: List[str]
    ) -> Dataset:
        """
        Prepare dataset for few-shot fine-tuning.
        
        Args:
            examples: List of examples with 'text' and 'label' keys
            candidate_labels: List of possible labels
            
        Returns:
            Hugging Face Dataset object
        """
        # Convert examples to dataset format
        texts = [ex['text'] for ex in examples]
        labels = [ex['label'] for ex in examples]
        
        # Create label mapping
        label2id = {label: idx for idx, label in enumerate(candidate_labels)}
        id2label = {idx: label for label, idx in label2id.items()}
        
        # Convert labels to IDs
        label_ids = [label2id[label] for label in labels]
        
        dataset = Dataset.from_dict({
            'text': texts,
            'labels': label_ids
        })
        
        # Store label mappings
        self.label2id = label2id
        self.id2label = id2label
        
        logger.info(f"Prepared dataset with {len(dataset)} examples and {len(candidate_labels)} labels")
        return dataset
    
    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """Tokenize the dataset for training."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding=True,
                max_length=self.config.max_length
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset
    
    def fine_tune_model(
        self, 
        train_dataset: Dataset, 
        eval_dataset: Optional[Dataset] = None,
        output_dir: str = "./models/fine_tuned"
    ) -> None:
        """
        Fine-tune the model on few-shot examples.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            output_dir: Directory to save the fine-tuned model
        """
        # Load model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Tokenize datasets
        train_dataset = self.tokenize_dataset(train_dataset)
        if eval_dataset:
            eval_dataset = self.tokenize_dataset(eval_dataset)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            learning_rate=self.config.learning_rate,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=50 if eval_dataset else None,
            save_strategy="steps",
            save_steps=100,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Train the model
        logger.info("Starting fine-tuning...")
        trainer.train()
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Fine-tuning completed. Model saved to {output_dir}")
    
    def evaluate_model(
        self, 
        test_dataset: Dataset, 
        model_path: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate the fine-tuned model on test data.
        
        Args:
            test_dataset: Test dataset
            model_path: Path to the fine-tuned model (optional)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if model_path:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before evaluation")
        
        # Tokenize test dataset
        test_dataset = self.tokenize_dataset(test_dataset)
        
        # Get predictions
        predictions = []
        true_labels = []
        
        for example in test_dataset:
            inputs = self.tokenizer(
                example['text'],
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.config.max_length
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                prediction = torch.argmax(outputs.logits, dim=-1).item()
                predictions.append(prediction)
                true_labels.append(example['labels'])
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    def save_results(self, results: List[ClassificationResult], filepath: str) -> None:
        """Save classification results to JSON file."""
        results_dict = []
        for result in results:
            results_dict.append({
                'text': result.text,
                'predicted_label': result.predicted_label,
                'confidence': result.confidence,
                'all_scores': result.all_scores,
                'model_name': result.model_name
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str) -> List[ClassificationResult]:
        """Load classification results from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            results_dict = json.load(f)
        
        results = []
        for result_dict in results_dict:
            results.append(ClassificationResult(
                text=result_dict['text'],
                predicted_label=result_dict['predicted_label'],
                confidence=result_dict['confidence'],
                all_scores=result_dict['all_scores'],
                model_name=result_dict['model_name']
            ))
        
        logger.info(f"Loaded {len(results)} results from {filepath}")
        return results
