"""Setup configuration for ShiftBench."""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="shiftbench",
    version="1.0.0",
    author="ShiftBench Contributors",
    description="A Benchmark Suite for Distribution Shift Evaluation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anthropics/shift-bench",
    project_urls={
        "Bug Tracker": "https://github.com/anthropics/shift-bench/issues",
        "Documentation": "https://shift-bench.readthedocs.io",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "mypy>=0.950",
            "flake8>=4.0.0",
        ],
        "ravel": [
            # Dependency on RAVEL source for baseline implementation
            # Users should install RAVEL separately
        ],
        "chem": [
            "rdkit>=2022.03.1",
        ],
        "full": [
            "scikit-learn>=1.0.0",
            "torch>=1.12.0",
            "cvxpy>=1.2.0",  # For KMM
        ],
    },
)
