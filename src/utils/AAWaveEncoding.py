import sys
import numpy as np
import math

class AAWaveEncoding(object):
    def __init__(self) -> None:
        self.AA_dict = {aa:i for i, aa in enumerate("ARNDCQEGHILKMFPSTWYVUXO")}
        self.AA_ENCODED_MEMORY = {key:self.get(val, 20) for key, val in self.AA_dict.items()}

    def get(self, i, dim):
        enc = np.zeros(dim)
        div_term = np.exp(np.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        enc[0::2] = np.sin(np.array([i]) * div_term)
        enc[1::2] = np.cos(np.array([i]) * div_term)
        return enc
