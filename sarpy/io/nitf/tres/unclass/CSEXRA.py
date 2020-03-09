# -*- coding: utf-8 -*-

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class CSEXRAType(TREElement):
    def __init__(self, value):
        super(CSEXRAType, self).__init__()
        self.add_field('SENSOR', 's', 6, value)
        self.add_field('TIME_FIRST_LINE_IMAGE', 'd', 12, value)
        self.add_field('TIME_IMAGE_DURATION', 'd', 12, value)
        self.add_field('MAX_GSD', 'd', 5, value)
        self.add_field('ALONG_SCAN_GSD', 's', 5, value)
        self.add_field('CROSS_SCAN_GSD', 's', 5, value)
        self.add_field('GEO_MEAN_GSD', 's', 5, value)
        self.add_field('A_S_VERT_GSD', 's', 5, value)
        self.add_field('C_S_VERT_GSD', 's', 5, value)
        self.add_field('GEO_MEAN_VERT_GSD', 's', 5, value)
        self.add_field('GSD_BETA_ANGLE', 's', 5, value)
        self.add_field('DYNAMIC_RANGE', 'd', 5, value)
        self.add_field('NUM_LINES', 'd', 7, value)
        self.add_field('NUM_SAMPLES', 'd', 5, value)
        self.add_field('ANGLE_TO_NORTH', 'd', 7, value)
        self.add_field('OBLIQUITY_ANGLE', 'd', 6, value)
        self.add_field('AZ_OF_OBLIQUITY', 'd', 7, value)
        self.add_field('GRD_COVER', 'd', 1, value)
        self.add_field('SNOW_DEPTH_CAT', 'd', 1, value)
        self.add_field('SUN_AZIMUTH', 'd', 7, value)
        self.add_field('SUN_ELEVATION', 'd', 7, value)
        self.add_field('PREDICTED_NIIRS', 's', 3, value)
        self.add_field('CIRCL_ERR', 'd', 3, value)
        self.add_field('LINEAR_ERR', 'd', 3, value)


class CSEXRA(TREExtension):
    _tag_value = 'CSEXRA'
    _data_type = CSEXRAType
