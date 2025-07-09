from setuptools import setup, find_packages

setup(
    name="openseq",
    version="0.1.0",
    description="A Python library for fitting multiple sequence alignment models",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="OpenSeq Contributors", # Updated placeholder
    author_email="openseq@example.com", # Updated placeholder
    url="https://github.com/your-repo/openseq",  # Updated placeholder URL
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=[
        "jax>=0.4.20",   # More recent JAX version
        "jaxlib>=0.4.20", # Corresponding jaxlib
        "numpy>=1.22",    # More recent numpy
        "optax>=0.1.7",   # For optimizers
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            # Add other dev dependencies like linters, formatters here
            # "flake8",
            # "black",
            # "isort",
        ],
        "examples": [ # Dependencies needed to run all examples
            # "matplotlib", # If examples use plotting
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11", # Added 3.11
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Typing :: Typed",
    ],
    python_requires=">=3.8",
    include_package_data=True, # If you have non-python files in your package (e.g. data for tests)
)
