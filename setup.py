import os

import pkg_resources
from setuptools import setup, find_packages

setup(
    name="alpha_clip",
    py_modules=["alpha_clip"],
    version="1.0",
    description="",
    author="OpenAI&ZeyiSun",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    include_package_data=True,
    extras_require={'dev': ['pytest']},
)
