from setuptools import find_packages, setup

setup(
    name="metamon",
    version="1.0.0",
    author="Jake Grigsby",
    author_email="grigsby@cs.utexas.edu",
    license="MIT",
    packages=find_packages(include=["metamon"]),
    python_requires=">=3.10",
    install_requires=[
        "gymnasium>=0.26,<=0.29.1",
        "torch>=2.5",
        "numpy",
        "gin-config",
        "wandb",
        "einops",
        "tqdm",
        "accelerate>=1.0",
        "termcolor",
        "huggingface_hub",
        "poke-env @ git+https://github.com/jakegrigsby/poke-env.git",
        "amago @ git+https://github.com/ut-austin-rpl/amago.git",
    ],
)
