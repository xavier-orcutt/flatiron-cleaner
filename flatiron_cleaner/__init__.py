"""
FlatironCleaner

A Python package for cleaning and harmonizing Flatiron Health cancer data.
"""

__version__ = '0.1.8'

# Make key classes available at package level
from .urothelial import DataProcessorUrothelial
from .nsclc import DataProcessorNSCLC
from .colorectal import DataProcessorColorectal
from .breast import DataProcessorBreast
from .prostate import DataProcessorProstate
from .renal import DataProcessorRenal
from .melanoma import DataProcessorMelanoma
from .general import DataProcessorGeneral
from .merge_utils import merge_dataframes

# Define what gets imported with `from flatiron_cleaner import *`
__all__ = [
    'DataProcessorUrothelial',
    'DataProcessorNSCLC',
    'DataProcessorColorectal',
    'DataProcessorBreast',
    'DataProcessorProstate',
    'DataProcessorRenal',
    'DataProcessoreMelanoma',
    'DataProcessorGeneral',
    'merge_dataframes',
]