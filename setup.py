# setup.py

from setuptools import setup, find_packages

# Read the README.md file for the long description
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='process_improvement',
    version='1.1.8',
    packages=find_packages(),
    description='A Python library for performing calculations and generating figures that facilitate an understanding of process variation.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Jim Lehner',
    author_email='James.Lehner@gmail.com',
    url='https://github.com/jimlehner/process-improvement',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    license='MIT',  
    keywords='''business, process improvement, industrial, data-driven, process engineering, quality, manufacturing, automation, applied analytics, control chart, process improvement,
    process behavior chart, x-chart, mR-chart, XmR-chart, indivdual values chart, moving range chart, continuous improvement,
    Deming, Shewhart, healthcare, education, government''', 
    install_requires=[
        'pandas', 
        'numpy', 
        'matplotlib',
        'seaborn']
    )
