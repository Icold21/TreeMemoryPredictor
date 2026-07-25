from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tree_memory_predictor",
    version="1.0.0",
    author="Ivan Kholodilo",
    author_email="iholodilo2008@gmail.com",
    description="High-performance, adaptive sequence prediction engine based on Context Mixing and Variable Order Markov Models (VOMM).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Icold21/TreeMemoryPredictor",
    packages=find_packages(),
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8',
    install_requires=[
        "tqdm"
    ],
    extras_require={
        "full": [
            "numpy",
            "matplotlib",
            "tiktoken",
            "pypdf",
            "jupyter",
            "pytest>=7.0.0"
        ]
    }
)