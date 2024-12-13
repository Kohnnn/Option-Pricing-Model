from setuptools import setup, find_packages

setup(
    name="option-pricing-model",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.26.0',
        'scipy>=1.11.0',
        'pandas>=2.1.0',
        'matplotlib>=3.8.0',
        'flask>=2.3.2',
        'flask-cors>=4.0.0',
        'streamlit>=1.28.0',
        'plotly>=5.17.0',
        'scikit-learn>=1.3.0',
        'seaborn>=0.12.2',
    ],
    python_requires='>=3.9,<3.13',
)
