"""
FlatironCleaner

A Python package for cleaning and harmonizing Flatiron Health cancer data.
"""

__version__ = '0.1.12'

# Make key classes available at package level
from .general import DataProcessorGeneral
from .urothelial import DataProcessorUrothelial
from .nsclc import DataProcessorNSCLC
from .colorectal import DataProcessorColorectal
from .breast import DataProcessorBreast
from .prostate import DataProcessorProstate
from .renal import DataProcessorRenal
from .melanoma import DataProcessorMelanoma
from .headneck import DataProcessorHeadNeck
from .merge_utils import merge_dataframes

# Define what gets imported with `from flatiron_cleaner import *`
__all__ = [
    'DataProcessorGeneral',
    'DataProcessorUrothelial',
    'DataProcessorNSCLC',
    'DataProcessorColorectal',
    'DataProcessorBreast',
    'DataProcessorProstate',
    'DataProcessorRenal',
    'DataProcessorMelanoma',
    'DataProcessorHeadNeck'
    'merge_dataframes'
]