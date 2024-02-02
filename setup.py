from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='vertisim',
    version='1.0.0',
    # package_dir={"": "vertisim"},
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "geog==0.0.2",
        "geopandas>=0.9.0",
        "networkx>=2.6.2",
        "numpy>=1.21.2",
        "pandas>=1.3.2",
        "scipy>=1.7.1",
        "Shapely>=1.7.1",
        "simpy==4.0.1",
        "sympy>=1.11.1",
        "tqdm>=4.64.1",
        "xlrd>=2.0.1"
    ],
    python_requires=">=3.11",
    description='VertiSim is a multi-agent simulation environment for urban air mobility networks.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eminb61/VertiSim/",
    author='Emin Burak Onat',
    author_email='eminburak_onat@berkeley.edu',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.11',
    ],
)
