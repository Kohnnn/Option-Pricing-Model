from setuptools import setup, find_packages

setup(
    name="option-pricing-model",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy==1.24.3',
        'scipy==1.10.1',
        'pandas==2.0.1',
        'matplotlib==3.7.1',
        'flask==2.3.2',
        'flask-cors==4.0.0',
        'streamlit==1.22.0',
        'plotly==5.14.1',
        'scikit-learn==1.2.2',
        'seaborn==0.12.2',
    ],
    python_requires='>=3.8,<3.11',
)
