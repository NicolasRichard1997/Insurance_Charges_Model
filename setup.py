from os import path
from io import open
from setuptools import setup, find_packages

from insurance_charges_model import __name__, __version__, __doc__


def load_file(file_name):
    here = path.abspath(path.dirname(__file__))
    with open(path.join(here, file_name)) as f:
        return f.read()


setup(name=__name__,
      version=__version__,
      author="Brian Schmidt",
      author_email="6666331+schmidtbri@users.noreply.github.com",
      description=__doc__,
      long_description=load_file("README.md"),
      long_description_content_type="text/markdown",
      url="https://github.com/schmidtbri/regression-model",
      license="BSD",
      packages=find_packages(exclude=["tests", "*tests", "tests*"]),
      install_requires=["ml-base>=0.1.0", "rest_model_service>=0.1.0", "pandas", "scikit-learn==0.24.2",
                        "featuretools==0.24.0", "tpot==0.11.7"],
      extras_require={
            "training": ["kaggle", "jupyter", "pandas_profiling", "yellowbrick"],
      },
      tests_require=["pytest", "pytest-html", "pylama", "coverage", "coverage-badge", "bandit", "safety", "pytype"],
      package_data={
            "insurance_charges_model": [
                  "model_files/*/*.joblib",
                  "model_files/*/*.html",
                  "training/*.ipynb"
            ]
      },
      classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent"
      ],
      project_urls={
            "Documentation": "https://schmidtbri.github.io/regression-model",
            "Source Code": "https://github.com/schmidtbri/regression-model",
            "Tracker": "https://github.com/schmidtbri/regression-model/issues"
      },
      keywords=[
            "machine learning", "automated machine learning", "model deployment", "regression model"
      ])
