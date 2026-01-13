#!/usr/bin/env python3
"""
Setup script for Few-Shot Text Classification project

This script helps set up the project environment and dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported. Please use Python 3.8+")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def create_directories():
    """Create necessary directories."""
    print("📁 Creating project directories...")
    directories = [
        "data",
        "data/processed", 
        "models",
        "logs",
        "config",
        "tests",
        "web_app"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   Created: {directory}/")
    
    print("✅ All directories created")
    return True


def install_dependencies():
    """Install project dependencies."""
    print("📦 Installing dependencies...")
    
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing Python packages"):
        return False
    
    return True


def verify_installation():
    """Verify that key packages are installed."""
    print("🔍 Verifying installation...")
    
    packages_to_check = [
        "torch",
        "transformers", 
        "streamlit",
        "pandas",
        "numpy",
        "scikit-learn",
        "plotly"
    ]
    
    for package in packages_to_check:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} not found")
            return False
    
    print("✅ All packages verified")
    return True


def create_sample_data():
    """Create sample data for testing."""
    print("📊 Creating sample data...")
    
    try:
        from src.dataset_generator import SyntheticDatasetGenerator
        
        generator = SyntheticDatasetGenerator(seed=42)
        
        # Generate sample datasets
        datasets = {
            "news": generator.generate_news_dataset(100),
            "sentiment": generator.generate_sentiment_dataset(100),
            "topic": generator.generate_topic_dataset(100)
        }
        
        # Save datasets
        for dataset_type, dataset in datasets.items():
            generator.save_dataset(dataset, f"data/sample_{dataset_type}.json")
            print(f"   Created: data/sample_{dataset_type}.json ({len(dataset)} samples)")
        
        print("✅ Sample data created")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create sample data: {e}")
        return False


def run_tests():
    """Run the test suite."""
    print("🧪 Running tests...")
    
    if not Path("tests/test_few_shot_classifier.py").exists():
        print("⚠️  Test file not found, skipping tests")
        return True
    
    if not run_command("python -m pytest tests/ -v", "Running test suite"):
        print("⚠️  Some tests failed, but installation may still be working")
        return True  # Don't fail setup if tests fail
    
    return True


def main():
    """Main setup function."""
    print("🚀 FEW-SHOT TEXT CLASSIFICATION - SETUP")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        print("💡 Try running: pip install -r requirements.txt")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("❌ Installation verification failed")
        sys.exit(1)
    
    # Create sample data
    if not create_sample_data():
        print("⚠️  Failed to create sample data, but continuing...")
    
    # Run tests
    if not run_tests():
        print("⚠️  Tests failed, but setup completed")
    
    print("\n" + "=" * 50)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print("\n🚀 Next steps:")
    print("   1. Try the web interface:")
    print("      streamlit run web_app/app.py")
    print("\n   2. Use the command line interface:")
    print("      python cli.py interactive")
    print("\n   3. Run the example script:")
    print("      python example.py")
    print("\n   4. Explore the documentation:")
    print("      README.md")
    print("\n💡 For help, check the README.md file or open an issue on GitHub")


if __name__ == "__main__":
    main()
