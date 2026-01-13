"""
Configuration management for Few-Shot Text Classification

This module handles configuration loading and validation.
"""

import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    name: str = "facebook/bart-large-mnli"
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    device: str = "auto"


@dataclass
class DataConfig:
    """Data configuration parameters."""
    dataset_type: str = "news"  # news, sentiment, topic
    num_samples: int = 1000
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    data_dir: str = "data"
    output_dir: str = "data/processed"


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    output_dir: str = "models"
    logging_dir: str = "logs"
    save_steps: int = 100
    eval_steps: int = 50
    logging_steps: int = 10
    early_stopping_patience: int = 3
    gradient_accumulation_steps: int = 1


@dataclass
class AppConfig:
    """Application configuration parameters."""
    title: str = "Few-Shot Text Classification"
    description: str = "Interactive demo for few-shot learning"
    host: str = "localhost"
    port: int = 8501
    debug: bool = False


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    app: AppConfig
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create Config from dictionary."""
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            data=DataConfig(**config_dict.get('data', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            app=AppConfig(**config_dict.get('app', {}))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Config to dictionary."""
        return {
            'model': asdict(self.model),
            'data': asdict(self.data),
            'training': asdict(self.training),
            'app': asdict(self.app)
        }


class ConfigManager:
    """Manages configuration loading and saving."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or "config/config.yaml"
        self.config = None
    
    def load_config(self, config_path: Optional[str] = None) -> Config:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Loaded configuration
        """
        path = config_path or self.config_path
        path = Path(path)
        
        if not path.exists():
            logger.warning(f"Config file not found: {path}. Using default configuration.")
            self.config = Config(
                model=ModelConfig(),
                data=DataConfig(),
                training=TrainingConfig(),
                app=AppConfig()
            )
            return self.config
        
        try:
            if path.suffix == '.yaml' or path.suffix == '.yml':
                with open(path, 'r', encoding='utf-8') as f:
                    config_dict = yaml.safe_load(f)
            elif path.suffix == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {path.suffix}")
            
            self.config = Config.from_dict(config_dict)
            logger.info(f"Configuration loaded from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            logger.info("Using default configuration")
            self.config = Config(
                model=ModelConfig(),
                data=DataConfig(),
                training=TrainingConfig(),
                app=AppConfig()
            )
        
        return self.config
    
    def save_config(self, config: Config, config_path: Optional[str] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save
            config_path: Path to save configuration
        """
        path = config_path or self.config_path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            config_dict = config.to_dict()
            
            if path.suffix == '.yaml' or path.suffix == '.yml':
                with open(path, 'w', encoding='utf-8') as f:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif path.suffix == '.json':
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported config file format: {path.suffix}")
            
            logger.info(f"Configuration saved to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def get_config(self) -> Config:
        """Get current configuration."""
        if self.config is None:
            self.config = self.load_config()
        return self.config
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary with configuration updates
        """
        if self.config is None:
            self.config = self.load_config()
        
        # Update model config
        if 'model' in updates:
            model_dict = self.config.model.__dict__.copy()
            model_dict.update(updates['model'])
            self.config.model = ModelConfig(**model_dict)
        
        # Update data config
        if 'data' in updates:
            data_dict = self.config.data.__dict__.copy()
            data_dict.update(updates['data'])
            self.config.data = DataConfig(**data_dict)
        
        # Update training config
        if 'training' in updates:
            training_dict = self.config.training.__dict__.copy()
            training_dict.update(updates['training'])
            self.config.training = TrainingConfig(**training_dict)
        
        # Update app config
        if 'app' in updates:
            app_dict = self.config.app.__dict__.copy()
            app_dict.update(updates['app'])
            self.config.app = AppConfig(**app_dict)
        
        logger.info("Configuration updated")
    
    def create_default_config(self, config_path: Optional[str] = None) -> None:
        """
        Create default configuration file.
        
        Args:
            config_path: Path to save default configuration
        """
        default_config = Config(
            model=ModelConfig(),
            data=DataConfig(),
            training=TrainingConfig(),
            app=AppConfig()
        )
        
        self.save_config(default_config, config_path)
        logger.info(f"Default configuration created at {config_path or self.config_path}")
