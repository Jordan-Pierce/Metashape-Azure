import os
from setuptools import setup, find_packages

# Check if requirements.txt exists
assert os.path.exists("requirements.txt"), "ERROR: Cannot find requirements.txt"

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    required_packages = f.read().splitlines()

# Filter out any empty lines or comments
required_packages = [line for line in required_packages if line and not line.startswith('#')]
required_packages = [line for line in required_packages if not line.startswith("https://")]

# Setup
setup(
    name='metashape-azure-mls',
    version='0.0.1',
    description='A user interface for submitting Metashape jobs to Azure ML Studio, or locally.',
    url='https://github.com/Jordan-Pierce/Metashape-Azure',
    author='Jordan Pierce, Ben Wade',
    author_email='jordan.pierce@noaa.gov, ben.wade@noaa.gov',
    packages=find_packages(),
    install_requires=required_packages + [
        "metashape @ https://download.agisoft.com/Metashape-2.1.2-cp37.cp38.cp39.cp310.cp311-none-win_amd64.whl"
    ],
    python_requires='>=3.8',
    entry_points={
        "console_scripts": [
            "metashape-azure-mls = src:main_function"
        ]
    },
)