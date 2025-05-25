"""Setup script for phosphorylation prediction package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
with open('requirements.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            requirements.append(line)

setup(
    name="phosphorylation-prediction",
    version="1.0.0",
    author="Research Team",
    author_email="research@university.edu",
    description="A standardized framework for phosphorylation site prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/research-team/phosphorylation-prediction",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
            "mypy>=0.950",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
        ],
        "gpu": [
            "torch[gpu]>=1.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "phos-train=scripts.train:main",
            "phos-predict=scripts.predict:main",
            "phos-analyze=scripts.analyze:main",
        ],
    },
    include_package_data=True,
    package_data={
        "config": ["*.yaml"],
    },
    zip_safe=False,
    keywords="phosphorylation prediction machine-learning bioinformatics protein",
    project_urls={
        "Bug Reports": "https://github.com/research-team/phosphorylation-prediction/issues",
        "Source": "https://github.com/research-team/phosphorylation-prediction",
        "Documentation": "https://phosphorylation-prediction.readthedocs.io/",
    },
)