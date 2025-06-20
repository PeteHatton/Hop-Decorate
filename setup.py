import versioneer
from setuptools import setup, find_packages

setup(
    name="Hop-Decorate",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    description="High-throughput defect transition sampling for chemically complex materials",
    author="Peter Hatton",
    author_email="pete.hatton21@gmail.com",
    url="https://github.com/PeteHatton/Hop-Decorate/tree/main",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD-3 License",  # change if needed
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)