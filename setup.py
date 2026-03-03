"""
ECG Classification Project - Setup Configuration

This file contains setup configuration for installing the project as a package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ecg-classification",
    version="1.0.0",
    author="Mohamad AlJasem",
    author_email="mohamad@aljasem.eu.org",
    url="https://github.com/m-aljasem/ecg-arrhythmia-classifier-AI",
    description="Classification of Life-Threatening Arrhythmia ECG Signals Using Deep Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
