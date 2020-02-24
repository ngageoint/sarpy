from algorithm_toolkit import Algorithm, AlgorithmChain


class Main(Algorithm):

    def run(self):
        cl = self.cl  # type: AlgorithmChain.ChainLedger
        params = self.params  # type: dict
        # Add your algorithm code here

        ro = params['sarpy_reader']

        chain_output = {
            "output_type": "text",
            "output_value": str(ro.sicdmeta)
        }
        cl.add_to_metadata('chain_output_value', chain_output)

        # Do not edit below this line
        return cl
