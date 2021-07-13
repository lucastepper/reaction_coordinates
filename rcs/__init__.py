import numpy as np
from .hb4 import HB4


def load_xvg(file, max_rows=1000):
    """ Load a plumed .xvg file output. """
    output = None
    skiped_rows = 0
    while output is None:
        try:
            output = np.loadtxt(file, skiprows=skiped_rows)
        except ValueError:
            skiped_rows += 1
        if skiped_rows > max_rows:
            raise ValueError(f'Could not load {file}, the first {max_rows} lines were misformated')
    return output
