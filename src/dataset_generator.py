"""
Synthetic Dataset Generator for Few-Shot Text Classification

This module generates synthetic datasets for testing and demonstration purposes.
"""

import json
import random
from typing import List, Dict, Any, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SyntheticDatasetGenerator:
    """Generator for creating synthetic text classification datasets."""
    
    def __init__(self, seed: int = 42):
        """Initialize the generator with a random seed."""
        random.seed(seed)
        self.seed = seed
    
    def generate_news_dataset(self, num_samples: int = 1000) -> List[Dict[str, Any]]:
        """
        Generate a synthetic news classification dataset.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            List of examples with 'text' and 'label' keys
        """
        categories = {
            'technology': [
                "New AI breakthrough in machine learning research",
                "Tech company launches innovative software solution",
                "Digital transformation accelerates in businesses",
                "Cybersecurity threats increase globally",
                "Cloud computing adoption reaches new heights",
                "Mobile app development trends for 2024",
                "Artificial intelligence revolutionizes healthcare",
                "Blockchain technology gains mainstream adoption",
                "Internet of Things devices proliferate",
                "Quantum computing advances accelerate"
            ],
            'healthcare': [
                "New medical treatment shows promising results",
                "Healthcare system faces staffing challenges",
                "Mental health awareness campaigns expand",
                "Pharmaceutical research discovers new drug",
                "Telemedicine adoption increases significantly",
                "Public health initiatives improve outcomes",
                "Medical technology advances patient care",
                "Healthcare costs continue to rise",
                "Preventive medicine gains importance",
                "Global health crisis requires coordinated response"
            ],
            'finance': [
                "Stock market reaches new all-time high",
                "Central bank announces interest rate changes",
                "Cryptocurrency market experiences volatility",
                "Banking sector implements new regulations",
                "Investment strategies adapt to market conditions",
                "Financial technology disrupts traditional banking",
                "Economic indicators show mixed signals",
                "Corporate earnings exceed expectations",
                "International trade agreements impact markets",
                "Personal finance education gains importance"
            ],
            'politics': [
                "Government announces new policy initiatives",
                "Election campaigns intensify across regions",
                "International diplomatic relations evolve",
                "Legislative body passes important bills",
                "Political parties prepare for upcoming elections",
                "Public opinion polls show changing trends",
                "Government transparency initiatives expand",
                "International cooperation agreements signed",
                "Political debates focus on key issues",
                "Civic engagement increases among citizens"
            ],
            'sports': [
                "Championship game draws record audience",
                "Athlete breaks long-standing world record",
                "Sports league announces rule changes",
                "Olympic preparations accelerate globally",
                "Professional team signs star player",
                "Youth sports participation increases",
                "Sports technology enhances performance",
                "International tournament begins",
                "Athletic training methods evolve",
                "Sports broadcasting reaches new platforms"
            ]
        }
        
        dataset = []
        for _ in range(num_samples):
            category = random.choice(list(categories.keys()))
            template = random.choice(categories[category])
            
            # Add some variation to the text
            variations = [
                template,
                f"Breaking: {template.lower()}",
                f"Latest update: {template.lower()}",
                f"Report: {template.lower()}",
                f"Analysis: {template.lower()}",
                template.replace("New", "Recent").replace("new", "recent"),
                template.replace("announces", "reveals").replace("Announces", "Reveals"),
                template.replace("increases", "grows").replace("Increases", "Grows")
            ]
            
            text = random.choice(variations)
            dataset.append({'text': text, 'label': category})
        
        logger.info(f"Generated {num_samples} news samples across {len(categories)} categories")
        return dataset
    
    def generate_sentiment_dataset(self, num_samples: int = 1000) -> List[Dict[str, Any]]:
        """
        Generate a synthetic sentiment analysis dataset.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            List of examples with 'text' and 'label' keys
        """
        sentiments = {
            'positive': [
                "This product is absolutely amazing and exceeded my expectations",
                "I love how easy it is to use this service",
                "Outstanding quality and excellent customer support",
                "Highly recommend this to everyone",
                "Fantastic experience from start to finish",
                "This is exactly what I was looking for",
                "Perfect solution for my needs",
                "Couldn't be happier with this purchase",
                "Excellent value for money",
                "Top-notch service and quality"
            ],
            'negative': [
                "This is the worst product I've ever used",
                "Terrible customer service and poor quality",
                "Complete waste of money and time",
                "I regret buying this product",
                "Disappointed with the overall experience",
                "This doesn't work as advertised",
                "Poor quality and unreliable service",
                "Not worth the money at all",
                "Frustrated with the constant issues",
                "Would not recommend to anyone"
            ],
            'neutral': [
                "The product arrived on time as expected",
                "Standard quality for the price point",
                "It works fine but nothing special",
                "Average experience overall",
                "The service meets basic requirements",
                "Decent product with room for improvement",
                "It does what it's supposed to do",
                "Fair quality for the price",
                "Standard delivery and packaging",
                "Meets expectations without exceeding them"
            ]
        }
        
        dataset = []
        for _ in range(num_samples):
            sentiment = random.choice(list(sentiments.keys()))
            template = random.choice(sentiments[sentiment])
            
            # Add some variation
            variations = [
                template,
                f"Review: {template.lower()}",
                f"My experience: {template.lower()}",
                f"Feedback: {template.lower()}",
                template.replace("This", "It").replace("this", "it"),
                template.replace("I", "We").replace("i", "we"),
                template.replace("amazing", "incredible").replace("Amazing", "Incredible"),
                template.replace("terrible", "awful").replace("Terrible", "Awful")
            ]
            
            text = random.choice(variations)
            dataset.append({'text': text, 'label': sentiment})
        
        logger.info(f"Generated {num_samples} sentiment samples across {len(sentiments)} categories")
        return dataset
    
    def generate_topic_dataset(self, num_samples: int = 1000) -> List[Dict[str, Any]]:
        """
        Generate a synthetic topic classification dataset.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            List of examples with 'text' and 'label' keys
        """
        topics = {
            'education': [
                "Students learn new programming languages in computer science",
                "University research advances scientific understanding",
                "Online learning platforms expand educational access",
                "Teachers adapt to new teaching methodologies",
                "Educational technology transforms classroom experience",
                "Scholarship programs support student success",
                "Curriculum development focuses on practical skills",
                "Academic institutions embrace digital transformation",
                "Student engagement increases with interactive learning",
                "Educational assessment methods evolve"
            ],
            'entertainment': [
                "New movie releases attract large audiences",
                "Streaming platforms compete for subscribers",
                "Music festivals return after pandemic hiatus",
                "Gaming industry experiences record growth",
                "Celebrity news dominates social media",
                "Theater productions adapt to digital formats",
                "Podcast popularity continues to rise",
                "Virtual reality entertainment gains traction",
                "Comedy shows provide relief during difficult times",
                "Entertainment industry embraces new technologies"
            ],
            'travel': [
                "Tourism industry recovers from recent challenges",
                "Travel restrictions ease for international visitors",
                "Eco-tourism gains popularity among travelers",
                "Digital nomad lifestyle becomes mainstream",
                "Adventure travel experiences attract thrill-seekers",
                "Cultural tourism promotes local communities",
                "Sustainable travel practices gain importance",
                "Travel technology simplifies trip planning",
                "Domestic tourism increases significantly",
                "Travel insurance becomes essential for trips"
            ],
            'food': [
                "Restaurant industry adapts to changing consumer habits",
                "Sustainable food practices gain momentum",
                "Local cuisine attracts food enthusiasts",
                "Food delivery services expand rapidly",
                "Plant-based diets become more popular",
                "Cooking shows inspire home chefs",
                "Food waste reduction initiatives grow",
                "Farm-to-table movement gains support",
                "International cuisine influences local menus",
                "Food technology improves production efficiency"
            ],
            'environment': [
                "Climate change initiatives gain global support",
                "Renewable energy adoption accelerates worldwide",
                "Environmental conservation efforts expand",
                "Sustainable living practices become mainstream",
                "Carbon footprint reduction becomes priority",
                "Green technology investments increase",
                "Ecosystem protection measures strengthen",
                "Environmental education programs grow",
                "Clean energy solutions advance rapidly",
                "Biodiversity conservation gains importance"
            ]
        }
        
        dataset = []
        for _ in range(num_samples):
            topic = random.choice(list(topics.keys()))
            template = random.choice(topics[topic])
            
            # Add variation
            variations = [
                template,
                f"Report: {template.lower()}",
                f"News: {template.lower()}",
                f"Update: {template.lower()}",
                template.replace("New", "Recent").replace("new", "recent"),
                template.replace("gains", "achieves").replace("Gains", "Achieves"),
                template.replace("increases", "grows").replace("Increases", "Grows"),
                template.replace("becomes", "turns into").replace("Becomes", "Turns into")
            ]
            
            text = random.choice(variations)
            dataset.append({'text': text, 'label': topic})
        
        logger.info(f"Generated {num_samples} topic samples across {len(topics)} categories")
        return dataset
    
    def split_dataset(
        self, 
        dataset: List[Dict[str, Any]], 
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            dataset: Full dataset to split
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            
        Returns:
            Tuple of (train, validation, test) datasets
        """
        random.shuffle(dataset)
        
        total_size = len(dataset)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        
        train_dataset = dataset[:train_size]
        val_dataset = dataset[train_size:train_size + val_size]
        test_dataset = dataset[train_size + val_size:]
        
        logger.info(f"Split dataset: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
        return train_dataset, val_dataset, test_dataset
    
    def save_dataset(
        self, 
        dataset: List[Dict[str, Any]], 
        filepath: str,
        format: str = "json"
    ) -> None:
        """
        Save dataset to file.
        
        Args:
            dataset: Dataset to save
            filepath: Path to save the file
            format: File format ('json' or 'csv')
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
        elif format == "csv":
            import pandas as pd
            df = pd.DataFrame(dataset)
            df.to_csv(filepath, index=False, encoding='utf-8')
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Dataset saved to {filepath} ({len(dataset)} samples)")
    
    def load_dataset(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Load dataset from file.
        
        Args:
            filepath: Path to the dataset file
            
        Returns:
            Loaded dataset
        """
        path = Path(filepath)
        
        if path.suffix == '.json':
            with open(filepath, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
        elif path.suffix == '.csv':
            import pandas as pd
            df = pd.read_csv(filepath)
            dataset = df.to_dict('records')
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        logger.info(f"Dataset loaded from {filepath} ({len(dataset)} samples)")
        return dataset
