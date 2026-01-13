"""
Test suite for Few-Shot Text Classification

This module contains unit tests for the few-shot text classification system.
"""

import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.few_shot_classifier import FewShotTextClassifier, FewShotConfig, ClassificationResult
from src.dataset_generator import SyntheticDatasetGenerator
from src.config_manager import ConfigManager, Config


class TestFewShotTextClassifier(unittest.TestCase):
    """Test cases for FewShotTextClassifier."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = FewShotConfig(
            model_name="facebook/bart-large-mnli",
            max_length=128,
            batch_size=8
        )
        self.classifier = FewShotTextClassifier(self.config)
    
    def test_initialization(self):
        """Test classifier initialization."""
        self.assertEqual(self.classifier.config.model_name, "facebook/bart-large-mnli")
        self.assertEqual(self.classifier.config.max_length, 128)
        self.assertEqual(self.classifier.config.batch_size, 8)
    
    def test_device_detection(self):
        """Test device detection."""
        device = self.classifier._get_device()
        self.assertIn(device, ["cpu", "cuda", "mps"])
    
    @patch('src.few_shot_classifier.pipeline')
    def test_load_zero_shot_classifier(self, mock_pipeline):
        """Test zero-shot classifier loading."""
        mock_classifier = MagicMock()
        mock_pipeline.return_value = mock_classifier
        
        self.classifier.load_zero_shot_classifier()
        
        mock_pipeline.assert_called_once()
        self.assertEqual(self.classifier.classifier, mock_classifier)
    
    def test_prepare_few_shot_dataset(self):
        """Test dataset preparation."""
        examples = [
            {'text': 'This is about technology', 'label': 'technology'},
            {'text': 'This is about healthcare', 'label': 'healthcare'},
            {'text': 'This is about finance', 'label': 'finance'}
        ]
        labels = ['technology', 'healthcare', 'finance']
        
        dataset = self.classifier.prepare_few_shot_dataset(examples, labels)
        
        self.assertEqual(len(dataset), 3)
        self.assertEqual(len(self.classifier.label2id), 3)
        self.assertEqual(len(self.classifier.id2label), 3)
    
    def test_save_and_load_results(self):
        """Test saving and loading results."""
        results = [
            ClassificationResult(
                text="Test text",
                predicted_label="technology",
                confidence=0.95,
                all_scores={"technology": 0.95, "healthcare": 0.05},
                model_name="test-model"
            )
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            self.classifier.save_results(results, temp_path)
            loaded_results = self.classifier.load_results(temp_path)
            
            self.assertEqual(len(loaded_results), 1)
            self.assertEqual(loaded_results[0].text, "Test text")
            self.assertEqual(loaded_results[0].predicted_label, "technology")
            self.assertEqual(loaded_results[0].confidence, 0.95)
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestSyntheticDatasetGenerator(unittest.TestCase):
    """Test cases for SyntheticDatasetGenerator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = SyntheticDatasetGenerator(seed=42)
    
    def test_initialization(self):
        """Test generator initialization."""
        self.assertEqual(self.generator.seed, 42)
    
    def test_generate_news_dataset(self):
        """Test news dataset generation."""
        dataset = self.generator.generate_news_dataset(100)
        
        self.assertEqual(len(dataset), 100)
        self.assertTrue(all('text' in ex and 'label' in ex for ex in dataset))
        
        labels = set(ex['label'] for ex in dataset)
        expected_labels = {'technology', 'healthcare', 'finance', 'politics', 'sports'}
        self.assertEqual(labels, expected_labels)
    
    def test_generate_sentiment_dataset(self):
        """Test sentiment dataset generation."""
        dataset = self.generator.generate_sentiment_dataset(100)
        
        self.assertEqual(len(dataset), 100)
        
        labels = set(ex['label'] for ex in dataset)
        expected_labels = {'positive', 'negative', 'neutral'}
        self.assertEqual(labels, expected_labels)
    
    def test_generate_topic_dataset(self):
        """Test topic dataset generation."""
        dataset = self.generator.generate_topic_dataset(100)
        
        self.assertEqual(len(dataset), 100)
        
        labels = set(ex['label'] for ex in dataset)
        expected_labels = {'education', 'entertainment', 'travel', 'food', 'environment'}
        self.assertEqual(labels, expected_labels)
    
    def test_split_dataset(self):
        """Test dataset splitting."""
        dataset = self.generator.generate_news_dataset(100)
        train, val, test = self.generator.split_dataset(dataset, 0.7, 0.15, 0.15)
        
        self.assertEqual(len(train), 70)
        self.assertEqual(len(val), 15)
        self.assertEqual(len(test), 15)
        self.assertEqual(len(train) + len(val) + len(test), 100)
    
    def test_save_and_load_dataset(self):
        """Test saving and loading datasets."""
        dataset = self.generator.generate_news_dataset(10)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            self.generator.save_dataset(dataset, temp_path)
            loaded_dataset = self.generator.load_dataset(temp_path)
            
            self.assertEqual(len(loaded_dataset), 10)
            self.assertEqual(loaded_dataset[0]['text'], dataset[0]['text'])
            self.assertEqual(loaded_dataset[0]['label'], dataset[0]['label'])
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestConfigManager(unittest.TestCase):
    """Test cases for ConfigManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config_manager = ConfigManager()
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = self.config_manager.get_config()
        
        self.assertEqual(config.model.name, "facebook/bart-large-mnli")
        self.assertEqual(config.data.dataset_type, "news")
        self.assertEqual(config.training.output_dir, "models")
        self.assertEqual(config.app.title, "Few-Shot Text Classification")
    
    def test_config_updates(self):
        """Test configuration updates."""
        updates = {
            'model': {'name': 'test-model', 'max_length': 256},
            'data': {'num_samples': 2000}
        }
        
        self.config_manager.update_config(updates)
        config = self.config_manager.get_config()
        
        self.assertEqual(config.model.name, "test-model")
        self.assertEqual(config.model.max_length, 256)
        self.assertEqual(config.data.num_samples, 2000)
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            config = self.config_manager.get_config()
            self.config_manager.save_config(config, temp_path)
            
            new_config_manager = ConfigManager(temp_path)
            loaded_config = new_config_manager.load_config()
            
            self.assertEqual(loaded_config.model.name, config.model.name)
            self.assertEqual(loaded_config.data.dataset_type, config.data.dataset_type)
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestClassificationResult(unittest.TestCase):
    """Test cases for ClassificationResult."""
    
    def test_classification_result_creation(self):
        """Test ClassificationResult creation."""
        result = ClassificationResult(
            text="Test text",
            predicted_label="technology",
            confidence=0.95,
            all_scores={"technology": 0.95, "healthcare": 0.05},
            model_name="test-model"
        )
        
        self.assertEqual(result.text, "Test text")
        self.assertEqual(result.predicted_label, "technology")
        self.assertEqual(result.confidence, 0.95)
        self.assertEqual(result.all_scores["technology"], 0.95)
        self.assertEqual(result.model_name, "test-model")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
