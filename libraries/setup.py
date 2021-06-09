import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gainet",
    version="0.0.1",
    author="Hector Otero; Rick Mulder; David Strömback",
    author_email="7hector2@gmail.com",
    description="A package to train a GAN in audio inpainting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iamhectorotero/mlp-final",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
