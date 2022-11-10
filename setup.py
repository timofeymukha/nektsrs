from setuptools import setup
from setuptools import find_packages

setup(
    name="nektsrs",
    version="0.0.1",
    description="Package for working with KTH Framework timeseries data from Nek5000",
    url="https://github.com/timofeymukha/nektsrs",
    author="Timofey Mukha",
    packages=find_packages(),
    entry_points={"console_scripts": []},
    license="MIT Licence",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT Licence",
    ],
    zip_safe=False,
)
