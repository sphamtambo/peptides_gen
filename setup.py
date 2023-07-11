from setuptools import find_packages, setup

with open("README.md", "r") as f:
    readme = f.read()

with open("requirements.txt", "r") as f:
    reqs = f.read().split("\n")

setup(
    name="rnn_peptides",
    version="1.0.0",
    author="Sphamandla Mtambo",
    author_email="sphamtambo@gmail.com",
    description="Antimicrobial peptide generation using RNN(LSTM)",
    long_description=readme,
    url="https://github.com/sphamtambo/rnn_peptides",
    license="MIT",
    packages=find_packages(),
    install_requires=reqs,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
