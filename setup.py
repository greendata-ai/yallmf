from setuptools import setup, find_packages

setup(
    name="yallmf",
    version="0.1.0",
    author="Kyle Napierkowski",
    author_email="kyle@kaleidoscopedata.com",
    description="Y'allMF - Yet Another Large Language Model Framework",
    packages=find_packages(),
    install_requires=[
        "requests",
        "numpy",
        "openai",
        "tiktoken"
    ],
)