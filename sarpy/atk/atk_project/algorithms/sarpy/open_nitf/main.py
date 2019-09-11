from algorithm_toolkit import Algorithm, AlgorithmChain

import sarpy.io.complex as cf
from sarpy import sarpy_support_dir
import os


class Main(Algorithm):

    def run(self):
        cl = self.cl  # type: AlgorithmChain.ChainLedger
        params = self.params  # type: dict
        # Add your algorithm code here

        # Open file
        fname = params['filename']
        ro = cf.open(fname)

        cl.add_to_metadata('sarpy_reader', ro)

        # Do not edit below this line
        return cl
