from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()
    setup(
        name="guv-calcs",
        url="https://github.com/jvbelenky/guv-calcs",
        version="0.0.15",
        author="J. Vivian Belenky",
        author_email="j.vivian.belenky@outlook.com",
        description="A library for carrying out fluence and irradiance calculations for germicidal UV (GUV) applications.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
            "License :: OSI Approved :: MIT License",
        ],
        packages=find_packages('src'),
        package_dir={'': 'src'},
        zip_safe=True,
        python_requires=">=3.8",
        install_requires=[
            "numpy",
            "scipy",
            "matplotlib",
            "plotly",
            "photompy"
        ],
    )
