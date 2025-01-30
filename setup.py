from setuptools import setup, find_packages

setup(
    name="purelab",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'cleanlab>=2.5.0',
        'pandas>=1.3.0',
        'numpy>=1.20.0',
        'scikit-learn>=1.0.0',
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A tool for data quality analysis built on top of Cleanlab",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/purelab",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
) 