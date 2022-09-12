#!/usr/bin/python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
import re

version = "1.1.0"

setup(
    name="idtrackerai-app",
    version="1.1.0",
    description="""""",
    author=["Jordi Torrents", "Ricardo Ribeiro", "Francisco Romero Ferrero"],
    author_email="idtrackerai@gmail.com",
    url="https://idtrackerai-app.readthedocs.org",
    packages=find_packages(),
    install_requires=[
        "rich",
    ],
    entry_points={
        "console_scripts": [
            "idtrackerai=idtrackerai_app.__main__:start",
        ],
    },
)
