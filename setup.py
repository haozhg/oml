from setuptools import setup, find_namespace_packages

# read readme as long description
with open("README.md", "r") as f:
    long_description = f.read()

# This call to setup() does all the work
setup(
    name="ai4sci.oml",
    version="0.2.2",
    description="AI4Science: Efficient data-driven Online Model Learning (OML) / system identification and control",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/haozhg/oml",
    author="Hao Zhang",
    author_email="haozhang@alumni.princeton.edu",
    license="MIT",
    keywords=[
        "machine-learning",
        "AI for Science",
        "data-driven modeling",
        "reduced-order-modeling",
        "dynamical systems",
        "control theory",
        "system identification",
        "online model learning",
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    python_requires='>=3.7',
    packages=find_namespace_packages(include=["ai4sci.*"]),
    include_package_data=False,
    install_requires=["numpy"],
)
