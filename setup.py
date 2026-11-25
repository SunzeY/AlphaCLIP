import os

from setuptools import setup, find_packages

setup(
    name="alpha_clip",
    py_modules=["alpha_clip"],
    version="1.0",
    description="",
    author="OpenAI&ZeyiSun",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        line.strip()
        for line in open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        if line.strip() and not line.startswith("#")
    ],
    include_package_data=True,
    extras_require={'dev': ['pytest']},
)
