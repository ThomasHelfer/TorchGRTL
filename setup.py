from setuptools import setup
from setuptools import find_packages


def readme():
    with open("README.md") as f:
        return f.read()


setup(
    name="GeneralRelativity",
    version="0.1",
    description="HBA with GPR and PCA",
    long_description=readme(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Topic :: Machine learning :: Physics :: Simulation :: General Relativity",
    ],
    keywords="Machine learning, Physics, Simulation, General Relativity",
    author="ThomasHelfer",
    author_email="thomashelfer@live.de",
    license="MIT",
    packages=find_packages(exclude=["tests"]),
    install_requires=["torch", "black", "pre-commit", "pytest", "numpy"],
    python_requires=">=3.5 ",
    include_package_data=True,
    zip_safe=False,
)
