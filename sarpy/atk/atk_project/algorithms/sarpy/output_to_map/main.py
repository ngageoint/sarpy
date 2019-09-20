from algorithm_toolkit import Algorithm, AlgorithmChain

from sarpy.geometry.point_projection import image_to_ground
from sarpy.geometry.geocoords import ecf_to_geodetic
from sarpy.visualization.remap import density
import numpy as np
from scipy.interpolate import griddata
import imageio
import base64
import os
import matplotlib.pyplot as plt
from pyproj import Proj, transform


def create_ground_grid(min_x,   # type: float
                       max_x,   # type: float
                       min_y,   # type: float
                       max_y,   # type: float
                       npix_x,  # type: int
                       npix_y,  # type: int
                       ):       # type: (...) -> (ndarray, ndarray)
    ground_y_arr, ground_x_arr = np.mgrid[0:npix_y, 0:npix_x]
    ground_x_arr = ground_x_arr/npix_x*(max_x - min_x)
    ground_y_arr = (ground_y_arr - npix_y) * -1
    ground_y_arr = ground_y_arr/npix_y*(max_y - min_y)
    ground_x_arr = ground_x_arr + min_x
    ground_y_arr = ground_y_arr + min_y
    x_gsd = np.abs(ground_x_arr[0, 1] - ground_x_arr[0, 0])
    y_gsd = np.abs(ground_y_arr[0, 0] - ground_y_arr[1, 0])
    return ground_x_arr + x_gsd/2.0, ground_y_arr - y_gsd/2.0


def sarpy2ortho(ro, decimation=10):

    nx = ro.sicdmeta.ImageData.FullImage.NumCols
    ny = ro.sicdmeta.ImageData.FullImage.NumRows

    xv, yv = np.meshgrid(range(nx), range(ny))
    xv = xv[::decimation, ::decimation]
    yv = yv[::decimation, ::decimation]
    npix = xv.size

    xv = np.reshape(xv, (npix,1))
    yv = np.reshape(yv, (npix,1))
    im_points = np.concatenate([xv,yv], axis=1)

    ecf = image_to_ground(im_points, ro.sicdmeta)

    minx = np.min(ecf[:,0])
    maxx = np.max(ecf[:,0])
    miny = np.min(ecf[:,1])
    maxy = np.max(ecf[:,1])
    meanz = np.mean(ecf[:,2])

    xi, yi = create_ground_grid(minx, maxx, miny, maxy, round(nx/decimation), round(ny/decimation))

    cdata = ro.read_chip[::decimation, ::decimation]
    pix = density(cdata)
    plt.figure()
    plt.imshow(pix, cmap='gray')  # Display subsampled image
    plt.show()

    pix = np.reshape(pix, npix)

    gridded = griddata(ecf[:,0:2], pix, (xi,yi), method='nearest').astype(np.uint8)
    plt.figure()
    plt.imshow(gridded, cmap='gray')  # Display subsampled image
    plt.show()

    # TODO handle different regions of globe
    p1 = Proj(init='epsg:4978')
    p2 = Proj(init='epsg:4326')

    # ulry, ulrx, ulrz = ecf_to_geodetic(minx, maxy, meanz)
    # lrry, lrrx, lrrz = ecf_to_geodetic(maxx, miny, meanz)

    # ul = [float(ulry[0]), float(ulrx[0])]
    # lr = [float(lrry[0]), float(lrrx[0])]
    ulr = transform(p1,p2,minx, maxy)
    lrr = transform(p1,p2,maxx, miny)

    extent = [ul,lr]

    return gridded, extent


class Main(Algorithm):

    def run(self):
        cl = self.cl  # type: AlgorithmChain.ChainLedger
        params = self.params  # type: dict
        # Add your algorithm code here

        ro = params['sarpy_reader']
        img, extent = sarpy2ortho(ro)
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
