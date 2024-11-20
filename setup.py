from pathlib import Path
import sys
from setuptools import setup

def find_requirements():
    requirements = []
    requirements_file = Path("requirements.txt")
    assert requirements_file.exists(), f"Not found option file: {requirements_file}"
    with open(requirements_file, "r", encoding="utf-8") as file:
        for line in file:
            requirements.append(line.strip())
    return requirements

setup(
    name='livecodebench',
    version='0.1.0',    
    description='Fork of Official Repository for Evaluation of LiveCodeBench',
    url='https://github.com/shuds13/pyexample',
    author='Naman Jain',
    author_email='naman1205jain@gmail.com',
    license='MIT',
    packages=['livecodebench'],
    install_requires=find_requirements(),
)