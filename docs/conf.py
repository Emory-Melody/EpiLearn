# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import importlib
import os
import sys

# Document this checkout, not a release installed from PyPI.
sys.path.insert(0, os.path.abspath('../'))

# epilearn.visualize imports matplotlib.pyplot at module level, and Read the Docs
# builders have no display.
os.environ.setdefault('MPLBACKEND', 'Agg')

# -- Import check ------------------------------------------------------------
#
# autodoc downgrades an ImportError to a warning and carries on, so one missing
# dependency publishes an API reference with whole pages empty instead of failing
# the build. That is how the 0.0.x docs lost most of their content: requirements.txt
# had no torch_geometric, so `import epilearn` aborted partway through the model
# zoo. Reproduced on this checkout with torch_geometric hidden: the build still
# exited 0, while API/tasks.html and API/visualization.html rendered zero
# signatures and API/models.html only 12 of 25 classes.
#
# Importing every documented subpackage here raises during conf.py execution
# instead, which aborts the build with a readable message.
#
# Add a module here when you add an autodoc directive for a new subpackage.
_DOCUMENTED_MODULES = [
    'epilearn',
    'epilearn.data',
    'epilearn.models.Temporal',
    'epilearn.models.Spatial',
    'epilearn.models.SpatialTemporal',
    'epilearn.models.General',
    'epilearn.tasks.forecast',
    'epilearn.tasks.detection',
    'epilearn.tasks.nowcast',
    'epilearn.tasks.scenario_modeling',
    'epilearn.utils.transforms',
    'epilearn.utils.metrics',
    'epilearn.utils.simulation',
    'epilearn.utils.uncertainty',
    'epilearn.utils.compartmental_models',
    'epilearn.visualize.plot',
    'epilearn.benchmark',
    'epilearn.strategies',
    'epilearn.ensemble',
    'epilearn.regime',
]


def _require_importable(module_names):
    """Abort the build if any documented module cannot be imported."""
    failures = []
    for name in module_names:
        try:
            importlib.import_module(name)
        except Exception as exc:                      # noqa: BLE001 - report anything
            failures.append(f'  {name}: {type(exc).__name__}: {exc}')
    if failures:
        raise RuntimeError(
            'Cannot import EpiLearn, so autodoc would silently publish an empty '
            'API reference. Install the documentation dependencies:\n'
            '  pip install -r docs/requirements_torch.txt -r docs/requirements.txt\n'
            'Failed imports:\n' + '\n'.join(failures)
        )


_require_importable(_DOCUMENTED_MODULES)

import epilearn  # noqa: E402 - must follow the import check above

# -- Project information -----------------------------------------------------

project = 'EpiLearn'
copyright = '2024, Melody Group'
author = 'Melody Group'

# Read the version from the package instead of hard-coding it, so the docs cannot
# drift from the code they document.
release = epilearn.__version__                  # e.g. '0.1.0'
version = '.'.join(release.split('.')[:2])      # e.g. '0.1'

# -- General configuration ---------------------------------------------------


extensions = [
    'sphinx.ext.mathjax',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
    'sphinx.ext.autosummary',
]


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Autodoc -----------------------------------------------------------------

# Foundation-model backends (`pip install epilearn[foundation]`) are imported
# lazily inside each model's _load_model(), so they are never needed to build the
# docs. Mocking them keeps them off the Read the Docs image: several are large and
# pull conflicting pins, and none of them is required to read a docstring.
autodoc_mock_imports = [
    'chronos',
    'uni2ts',
    'momentfm',
    'timesfm',
    'gluonts',
]

# nitpicky = True is deliberately OFF. Docstrings here name types informally
# ("torch.Tensor", "optional", "array-like") instead of as resolvable
# cross-references, so nitpicky mode adds ~450 warnings on this checkout (196 of
# them for torch.Tensor alone) and buries the ones that matter. It would also not
# have caught the failure this file guards against: a missing dependency shows up
# as an autodoc import warning, not as a broken reference. The import check above
# is the real safety net -- keep it.
nitpicky = False

# -- Options for HTML output -------------------------------------------------

# html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"
# html_static_path = ['_static']
