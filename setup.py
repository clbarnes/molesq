from pathlib import Path
from setuptools import setup, find_packages

with open(Path(__file__).resolve().parent / "README.md") as f:
    readme = f.read()

setup(
    name="molesq",
    url="https://github.com/clbarnes/molesq",
    author="Chris L. Barnes",
    description=(
        "Implementation of moving least squares for ND point and image deformation"
    ),
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["molesq"]),
    install_requires=[
        "numpy>=1.20",
        "scipy",
        # "backports.strenum; python_version < '3.10'"
    ],
    test_requires=["pytest", "pytest-benchmark"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7, <4.0",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
)
