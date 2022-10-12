import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

requires = [
    "matplotlib==3.5.1",
    "networkx==2.6.3"
    ]
    
setuptools.setup(
    name="aacbr",
    version="0.2.0",
    author="Guilherme Paulino-Passos",
    author_email="g.passos18@imperial.ac.uk",
    description="CLArg's basic implementation of AA-CBR.",
    long_description="file: README.md",
    long_description_content_type="text/markdown",
    url="https://github.com/CLArg-group/AACBR",
    project_urls={
        "Bug Tracker": "https://github.com/CLArg-group/AACBR/issues",
    },
    classifiers=[
        " Programming Language :: Python :: 3",
        " License :: OSI Approved :: MIT License",
        " Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],    
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=requires
)
