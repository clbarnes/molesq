from pathlib import Path
from setuptools import setup, find_packages

with open(Path(__file__).resolve().parent / "README.md") as f:
    readme = f.read()

setup(
    name="molesq",
    url="https://github.com/clbarnes/molesq",
    author="Chris L. Barnes",
    description="Implementation of moving least squares",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["molesq"]),
    install_requires=["numpy", "scipy", "backports.strenum; python_version < '3.10'"],
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
)
