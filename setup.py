import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fast-automl",
    version="0.0.1",
    author="Dillon Bowen",
    author_email="dsbowen@wharton.upenn.edu",
    description="ML for reasonable results in a reasonable timeframe",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://dsbowen.github.io/fast-automl/",
    packages=setuptools.find_packages(),
    install_requires=[
        'joblib>=1.0.0',
        'numpy>=1.20.0',
        'pandas>=1.2.1',
        'scikit-learn>=0.24.1',
        'scipy>=1.6.0',
        'xgboost>=1.3.3',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)