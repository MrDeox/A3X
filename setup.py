from setuptools import setup, find_packages

setup(
    name="a3x",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pytest>=8.3.5"
    ],
    python_requires=">=3.8"
) 