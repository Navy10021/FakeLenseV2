from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fakelensev2",
    version="2.0.0",
    author="Navy Lee",
    author_email="iyunseob4@gmail.com",
    description="An AI-Powered Fake News Detection System Integrating LLMs and Deep Reinforcement Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Navy10021/FakeLenseV2",
    packages=find_packages(include=["code", "code.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "fakelense-train=code.main:train",
            "fakelense-infer=code.main:infer_cli",
        ],
    },
)
