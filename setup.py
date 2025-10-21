"""
项目安装配置
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="multimodal-vlm",
    version="0.1.0",
    author="Multimodal VLM Team",
    author_email="example@email.com",
    description="A comprehensive multimodal vision-language model project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
    package_data={
        "multimodal_vlm": [
            "configs/*.py",
            "data/*.py",
            "models/*.py", 
            "training/*.py",
            "inference/*.py",
            "utils/*.py",
            "examples/*.py",
        ]
    },
    entry_points={
        "console_scripts": [
            "multimodal-demo=examples.web_demo:main",
            "multimodal-train=examples.fine_tuning_demo:main",
        ],
    },
)