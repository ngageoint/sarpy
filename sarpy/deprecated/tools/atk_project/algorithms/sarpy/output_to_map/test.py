from algorithm_toolkit import AlgorithmTestCase

from main import Main


class MainTestCase(AlgorithmTestCase):

    def runTest(self):
        # configure params for your algorithm
        self.params = {}

        self.alg = Main(cl=self.cl, params=self.params)
        self.alg.run()

        # Add tests and assertions below
