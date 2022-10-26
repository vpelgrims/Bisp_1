#from .BISP_1 import *

from .Bisp import *
from .Stars import *
from .Decomp import *
from .Priors import *

import os.path

here = os.path.dirname(__file__)

version = "unknown"

with open(here+'/../../pyproject.toml') as fp:
    for line in fp.read().splitlines():
        if line.startswith('version'):
            delim = '"' if '"' in line else "'"
            version = str(line.split(delim)[1])

__version__ = version
