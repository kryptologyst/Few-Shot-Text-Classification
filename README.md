# Few-Shot Text Classification

A comprehensive implementation of few-shot learning for text classification using state-of-the-art transformer models and Hugging Face libraries.

## Features

- **Zero-Shot Classification**: Classify text without training examples
- **Few-Shot Fine-tuning**: Train models with minimal labeled data
- **Multiple Model Support**: Compatible with various transformer models
- **Interactive Web Interface**: Streamlit-based demo application
- **Command Line Interface**: Full CLI for batch processing and automation
- **Synthetic Dataset Generation**: Create datasets for testing and demonstration
- **Comprehensive Configuration**: YAML-based configuration management
- **Type Hints & Documentation**: Modern Python with full type annotations
- **Unit Tests**: Comprehensive test suite for reliability

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.20+
- Streamlit (for web interface)
- Other dependencies listed in `requirements.txt`

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kryptologyst/Few-Shot-Text-Classification.git
   cd Few-Shot-Text-Classification
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Web Interface (Recommended)

Launch the interactive Streamlit application:

```bash
streamlit run web_app/app.py
```

Open your browser to `http://localhost:8501` and start classifying text!

### Command Line Interface

**Classify text:**
```bash
python cli.py classify --text "AI is revolutionizing healthcare" --labels "technology,healthcare,finance"
```

**Generate synthetic dataset:**
```bash
python cli.py generate --type news --samples 1000 --output-dir data/
```

**Train a model:**
```bash
python cli.py train --train-file data/train.json --val-file data/val.json --epochs 3
```

**Interactive mode:**
```bash
python cli.py interactive
```

### Python API

```python
from src.few_shot_classifier import FewShotTextClassifier, FewShotConfig

# Initialize classifier
config = FewShotConfig(model_name="facebook/bart-large-mnli")
classifier = FewShotTextClassifier(config)
classifier.load_zero_shot_classifier()

# Classify text
result = classifier.classify_text(
    text="The company just launched a new AI product for healthcare.",
    candidate_labels=["technology", "healthcare", "finance", "politics"]
)

print(f"Predicted: {result.predicted_label}")
print(f"Confidence: {result.confidence:.3f}")
```

## 📁 Project Structure

```
few-shot-text-classification/
├── src/                          # Source code
│   ├── __init__.py
│   ├── few_shot_classifier.py    # Main classifier implementation
│   ├── dataset_generator.py      # Synthetic dataset generation
│   └── config_manager.py        # Configuration management
├── web_app/                      # Streamlit web application
│   └── app.py
├── tests/                        # Unit tests
│   └── test_few_shot_classifier.py
├── config/                       # Configuration files
│   └── config.yaml
├── data/                         # Data directory
├── models/                       # Trained models
├── logs/                         # Log files
├── cli.py                        # Command line interface
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## 🔧 Configuration

The application uses YAML configuration files. Key settings include:

- **Model**: Model name, batch size, learning rate, etc.
- **Data**: Dataset type, number of samples, train/val/test ratios
- **Training**: Output directories, logging settings, early stopping
- **App**: Web interface settings

Edit `config/config.yaml` to customize behavior.

## Supported Datasets

The system includes generators for three types of synthetic datasets:

1. **News Classification**: Technology, healthcare, finance, politics, sports
2. **Sentiment Analysis**: Positive, negative, neutral
3. **Topic Classification**: Education, entertainment, travel, food, environment

## Supported Models

- `facebook/bart-large-mnli` (default, recommended)
- `microsoft/DialoGPT-medium`
- `distilbert-base-uncased`
- `roberta-base`
- Any Hugging Face model compatible with zero-shot classification

## Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

Or run specific tests:

```bash
python tests/test_few_shot_classifier.py
```

## Performance

The system supports various evaluation metrics:

- **Accuracy**: Overall classification accuracy
- **Precision**: Precision for each class
- **Recall**: Recall for each class
- **F1 Score**: F1 score for each class
- **Confusion Matrix**: Detailed classification results

## Examples

### Zero-Shot Classification

```python
# Classify news articles
text = "Apple announces new AI-powered health monitoring features"
labels = ["technology", "healthcare", "finance", "politics"]
result = classifier.classify_text(text, labels)
# Result: technology (0.95 confidence)
```

### Few-Shot Training

```python
# Prepare few-shot examples
examples = [
    {"text": "New AI breakthrough", "label": "technology"},
    {"text": "Medical research advances", "label": "healthcare"},
    {"text": "Stock market rises", "label": "finance"}
]

# Train model
train_dataset = classifier.prepare_few_shot_dataset(examples, labels)
classifier.fine_tune_model(train_dataset)
```

### Batch Processing

```python
# Classify multiple texts
texts = [
    "AI revolutionizes healthcare",
    "Stock market reaches new highs",
    "New political policies announced"
]
results = classifier.batch_classify(texts, labels)
```

## Advanced Usage

### Custom Model Configuration

```python
config = FewShotConfig(
    model_name="custom-model",
    max_length=1024,
    batch_size=32,
    learning_rate=1e-5,
    num_epochs=5
)
```

### Dataset Customization

```python
# Generate custom dataset
generator = SyntheticDatasetGenerator(seed=123)
dataset = generator.generate_news_dataset(num_samples=5000)

# Split and save
train, val, test = generator.split_dataset(dataset, 0.8, 0.1, 0.1)
generator.save_dataset(train, "data/custom_train.json")
```

### Results Export

```python
# Save classification results
classifier.save_results(results, "output/results.json")

# Load previous results
previous_results = classifier.load_results("output/results.json")
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for the Transformers library
- [Streamlit](https://streamlit.io/) for the web interface framework
- [PyTorch](https://pytorch.org/) for the deep learning framework
- The open-source community for inspiration and contributions

## Support

For questions, issues, or contributions:

- Open an issue on GitHub
- Check the documentation in the `src/` directory
- Review the test cases in `tests/`
# Few-Shot-Text-Classification
