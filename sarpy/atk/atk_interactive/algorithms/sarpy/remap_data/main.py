from algorithm_toolkit import Algorithm, AlgorithmChain

from sarpy.visualization import remap


class Main(Algorithm):

    def run(self):
        cl = self.cl  # type: AlgorithmChain.ChainLedger
        params = self.params  # type: dict
        # Add your algorithm code here

        ro = params['sarpy_reader']
        decimation = params['decimation']

        if 'ystart' in params:
            ystart = params['ystart']
        else:
            ystart = 0

        if 'yend' in params:
            yend = params['yend']
        else:
            yend = ro.sicdmeta.ImageData.NumRows

        if 'xstart' in params:
            xstart = params['xstart']
        else:
            xstart = 0

        if 'xend' in params:
            xend = params['xend']
        else:
            xend = ro.sicdmeta.ImageData.NumCols

        remap_type = params['remap_type']

        cdata = ro.read_chip[ystart:yend:decimation, xstart:xend:decimation]

        if remap_type == 'density':
            pix = remap.density(cdata)
        elif remap_type == 'brighter':
            pix = remap.brighter(cdata)
        elif remap_type == 'darker':
            pix = remap.darker(cdata)
        elif remap_type == 'highcontrast':
            pix = remap.highcontrast(cdata)
        elif remap_type == 'linear':
            pix = remap.linear(cdata)
        elif remap_type == 'log':
            pix = remap.log(cdata)
        elif remap_type == 'pedf':
            pix = remap.pedf(cdata)
        elif remap_type == 'nrl':
            pix = remap.nrl(cdata)

        cl.add_to_metadata('remapped_data', pix)
        cl.add_to_metadata('decimation', decimation)

        # Do not edit below this line
        return cl
