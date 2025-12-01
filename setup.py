from setuptools import setup, find_packages

setup(
    name="corpus_processing_utils",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.0",
        "tqdm>=4.0",
        "openpyxl>=3.0",
        "jiwer>=2.3",
        "soundfile>=0.10"
    ],
    python_requires=">=3.8",
    author="asierhv",
    organization="HiTZ/Aholab - Basque Center for Language Technology",
    description="Utilities for audio/text corpus processing and WER calculation",
    url="https://github.com/asierhv/corpus_processing_utils",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
