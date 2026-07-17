"""Installation script for the 'luwu_mjlab' python package."""

from setuptools import setup, find_packages

# Pin the sim stack as one unit — mjlab dictates the mujoco/warp line
# (and rsl-rl-lib==5.4.0 transitively); bump all five together.
INSTALL_REQUIRES = [
    "mjlab==1.5.1",
    "mujoco==3.10.0",
    "mujoco-mjx==3.10.0",
    "mujoco-warp==3.10.0.2",
    "warp-lang==1.15.0",
    "scipy>=1.17.0",
]

# Installation operation
setup(
    name="luwu_mjlab",
    packages=["src"],
    version="1.0.0",
    install_requires=INSTALL_REQUIRES,
)
