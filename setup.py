from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "PosteriFlow: Adaptive Hierarchical Signal Decomposition for overlapping gravitational waves"

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, "r", encoding="utf-8") as fh:
        requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
else:
    requirements = [
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pycbc>=2.3.0",
        "lalsuite>=7.3",
        "gwpy>=3.0.0",
        "pandas>=2.0.0",
        "h5py>=3.8.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "scikit-learn>=1.2.0",
        "astropy>=5.2.0",
        "requests>=2.28.0",
    ]

setup(
    name="PosteriFlow",
    version="1.0.0",
    author="Bibin Thomas",
    author_email="bibinthomas951@gmail.com",
    description="Adaptive Hierarchical Signal Decomposition for overlapping gravitational waves",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bibinthomas123/PosteriFlow",
    packages=find_packages("src"),  # Look in src directory
    package_dir={"": "src"},  # Packages are under src
    package_data={
        "ahsd": [
            "*.yaml",
            "*.json",
            "configs/*.yaml",
            "data/configs/*.yaml",
            "data/psds/*.npz",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.900",
            "isort>=5.0",
            "pre-commit>=3.0",
            "jupyter>=1.0.0",
        ],
        "docs": [
            "sphinx>=6.0",
            "sphinx-rtd-theme>=1.2",
            "sphinx-autodoc-typehints>=1.20",
        ],
    },
    entry_points={
        "console_scripts": [
            "ahsd-analyze=ahsd.experiments.real_data_pipeline:main",
            "ahsd-train=ahsd.experiments.train_neuralpe:main",
            "ahsd-test=ahsd.experiments.test_neuralpe:main",
            "ahsd-generate=ahsd.data.scripts.generate_dataset:main",
            "ahsd-validate=ahsd.data.scripts.validate_dataset:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
