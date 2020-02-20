from algorithm_toolkit import Algorithm, AlgorithmChain

from sarpy.geometry.point_projection import image_to_ground
from sarpy.geometry.geocoords import ecf_to_geodetic
from resippy.image_objects.earth_overhead.\
    earth_overhead_image_objects.geotiff.geotiff_image_factory import GeotiffImageFactory
from resippy.utils.photogrammetry_utils import world_poly_to_geo_t, create_ground_grid
from resippy.utils.geom_utils import bounds_2_shapely_polygon
from pyproj import Proj

import numpy as np
from scipy.interpolate import griddata
import imageio
import base64
import os


def sarpy2ortho(ro, pix, decimation=10):

    nx = ro.sicdmeta.ImageData.FullImage.NumCols
    ny = ro.sicdmeta.ImageData.FullImage.NumRows

    nx_dec = round(nx / decimation)
    ny_dec = round(ny / decimation)

    xv, yv = np.meshgrid(range(nx), range(ny))
    xv = xv[::decimation, ::decimation]
    yv = yv[::decimation, ::decimation]
    npix = xv.size

    xv = np.reshape(xv, (npix,1))
    yv = np.reshape(yv, (npix,1))
    im_points = np.concatenate([xv,yv], axis=1)

    ground_coords = image_to_ground(im_points, ro.sicdmeta)
    ground_coords = ecf_to_geodetic(ground_coords)

    minx = np.min(ground_coords[:,1])
    maxx = np.max(ground_coords[:,1])
    miny = np.min(ground_coords[:,0])
    maxy = np.max(ground_coords[:,0])

    xi, yi = create_ground_grid(minx, maxx, miny, maxy, nx_dec, ny_dec)

    ground_coords[:,[0,1]] = ground_coords[:,[1,0]]
    pix = np.reshape(pix, npix)
    gridded = griddata(ground_coords[:,0:2], pix, (xi,yi), method='nearest').astype(np.uint8)

    ul = [maxy, minx]
    lr = [miny, maxx]

    extent = [ul,lr]

    extent_poly = bounds_2_shapely_polygon(min_x=minx, max_x=maxx, min_y=miny, max_y=maxy)
    geot = world_poly_to_geo_t(extent_poly, npix_x=nx_dec, npix_y=ny_dec)

    return gridded, extent, geot


class Main(Algorithm):

    def run(self):
        cl = self.cl  # type: AlgorithmChain.ChainLedger
        params = self.params  # type: dict
        # Add your algorithm code here

        ro = params['sarpy_reader']
        pix = params['remapped_data']
        decimation = params['decimation']

        output_path = None
        if 'geotiff_path' in params:
            output_path = os.path.expanduser(params['geotiff_path'])

        img, extent, geot = sarpy2ortho(ro, pix, decimation=decimation)

        if output_path is not None:
            fac = GeotiffImageFactory()
            geo_img = fac.from_numpy_array(img, geot, Proj(init='EPSG:4326'))
            geo_img.write_to_disk(output_path)

        im_pth = os.path.join(cl.get_temp_folder(), 'sar_ortho.png')
        imageio.imwrite(im_pth, img)

        with open(im_pth, 'rb') as fp:
            img_bytes = base64.b64encode(fp.read())
            img_str = img_bytes.decode('utf-8')

        chain_output = {
            "output_type": "geo_raster",
            "output_value": {
                "extent": str(extent),
                "raster": img_str
            }
        }
        cl.add_to_metadata('chain_output_value', chain_output)

        # Do not edit below this line
        return cl
