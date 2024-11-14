import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="epilearn",
    version="0.0.14",
    author="Emory-Melody", 
    maintainer='Emory-Melody',
    author_email="zevin.liu@gmail.com",
    description="A Pytorch library for machine learning in epidemic modeling", 
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Emory-Melody/EpiLearn",
    include_package_data=True,
    install_requires = [
          'matplotlib==3.9.1',
          'numpy==1.26.4',
          'scipy>=1.3.1',
          'networkx==3.2.1',
          'Pillow>=7.0.0',
          'scikit_learn>=0.22.1',
          'tqdm>=3.0',
          'seaborn>=0.13.2',
          'xgboost>=2.0.3',
          'statsmodels>=0.14.2',
          'streamlit==1.34.0',
          'pyvis==0.3.2',
          'plotly==5.22.0',
          'fastdtw==0.3.4',
          'einops==0.8.0' 
      ],
    packages=setuptools.find_packages(),
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries",
    ],
    license="MIT",
)