from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="ai-validation-framework",
    version="3.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Enterprise-grade ML validation pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/ai-validation-framework",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
    ],
    extras_require={
        "mlflow": ["mlflow>=2.8.0"],
        "api": ["fastapi>=0.104.0", "uvicorn[standard]>=0.24.0"],
        "viz": ["plotly>=5.18.0", "matplotlib>=3.8.0", "streamlit>=1.28.0"],
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0", "black>=23.0.0", "flake8>=6.0.0"],
        "all": [
            "mlflow>=2.8.0",
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.24.0",
            "plotly>=5.18.0",
            "matplotlib>=3.8.0",
            "streamlit>=1.28.0",
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ai-validate=ai_validation.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)