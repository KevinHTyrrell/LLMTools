from setuptools import setup, find_packages

setup(
    name='LLMTools',
    version='0.0.1',
    packages=find_packages(
        include=[
            'LLMTools'
        ]
    ),
    install_requires=[
        'faiss==1.5.3',
        'numpy==1.17.4',
        'openai==0.27.8',
        'pandas==0.25.3',
        'pypdf==3.14.0',
        'PyYAML==6.0.1',
        'scikit_learn==1.3.0',
        'sentence_transformers==2.2.2',
        'setuptools==45.2.0',
        'torch==2.0.1',
        'transformers==4.31.0'

    ]
)