from setuptools import find_packages, setup


def _fetch_requirements(path):
    with open(path, "r") as fd:
        return [r.strip() for r in fd.readlines()]


def _fetch_version():
    with open("version.txt", "r") as f:
        return f.read().strip()


setup(
    name="openrlhf",
    version=_fetch_version(),
    packages=find_packages(
        exclude=(
            "data",
            "docs",
            "examples",
        )
    ),
    description="A Ray-based High-performance RLHF framework.",
    install_requires=_fetch_requirements("requirements.txt"),
    extras_require={
        "vllm": ["vllm==0.11.0"],
        "ring": ["ring_flash_attn"],
        "liger": ["liger_kernel"],
    },
    python_requires=">=3.10",
)

