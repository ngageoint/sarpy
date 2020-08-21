# -*- coding: utf-8 -*-
"""
Module for reading and interpreting the standard change detection result files.
"""

import logging
import os
import json
from typing import Union, Dict

import numpy

from sarpy.compliance import int_func, string_types
from sarpy.io.general.base import AggregateReader
from sarpy.io.general.nitf import NITFReader, NITFWriter, ImageDetails, DESDetails, \
    image_segmentation, get_npp_block, interpolate_corner_points_string
from sarpy.io.general.nitf import NITFDetails
from sarpy.geometry.geometry_elements import FeatureCollection


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class ChangeDetectionDetails(object):
    """
    This is a helper class for interpreting the standard change detection results.
    """

    __slots__ = (
        '_file_names', '_features')

    def __init__(self, file_name):
        """

        Parameters
        ----------
        file_name : str
        """

        self._file_names = {}
        self._features = None

        if not isinstance(file_name, string_types):
            raise IOError('file_name is required to be a string path name.')
        if not os.path.isfile(file_name):
            raise IOError('file_name = {} is not a valid existing file'.format(file_name))

        root_dir, fname = os.path.split(file_name)
        fstem, fext = os.path.splitext(fname)
        if fext not in ['.ntf', '.nitf', '.json']:
            raise IOError('file_name is expected to have extension .ntf, .nitf, .json')
        if not (fstem.endswith('_C') or fstem.endswith('_M') or fstem.endswith('_R') or fstem.endswith('_V')):
            raise IOError(
                'file_name is expected to follow naming pattern XXX_C.ntf, XXX_M.ntf, XXX_R.ntf, '
                'or XXX_V.json')
        fstem = fstem[:-2]
        for part in ['C', 'M', 'R']:
            for ext in ['.ntf', '.nitf']:
                fil = os.path.join(root_dir, '{}_{}{}'.format(fstem, part, ext))
                if os.path.isfile(fil):
                    self._file_names[part] = fil
            if part not in self._file_names:
                raise IOError('Change detection file part {} not found'.format(part))
        fil = os.path.join(root_dir, '{}_V.json'.format(fstem))
        if os.path.isfile(fil):
            self._file_names['V'] = fil
        else:
            logging.warning('Change detection file part V (i.e. json metadata) not found.')

    @property
    def file_names(self):
        # type: () -> Dict[str, str]
        """
        Dict[str, str] : The file names.
        """

        return self._file_names

    @property
    def features(self):
        # type: () -> Union[FeatureCollection, None]
        """
        None|FeatureCollection: The feature list, if it is defined.
        """

        if 'V' in self._file_names and self._features is None:
            the_file = self._file_names['V']
            try:
                with open(the_file, 'r') as fi:
                    the_dict = json.load(fi)
                    self._features = FeatureCollection.from_dict(the_dict)
            except Exception as e:
                logging.error(
                    'Failed decoding change detection json file {}\n with error {}. '
                    'Skipping json definition.'.format(the_file, e))
                del self._file_names['V']  # drop this non-functional file to avoid repeating
        return self._features


class ChangeDetectionReader(AggregateReader):
    """
    An aggregate reader for the standard change detection scenario package files.
    The order of the aggregation is given by `C` (index 0), `M` (index 1), `R` (index 2).
    """

    __slots__ = ('_change_details', )

    def __init__(self, change_details):
        """

        Parameters
        ----------
        change_details : str|ChangeDetectionDetails
        """

        if isinstance(change_details, string_types):
            change_details = ChangeDetectionDetails(change_details)
        if not isinstance(change_details, ChangeDetectionDetails):
            raise TypeError(
                'change_details is required to be the file name of one of the change '
                'detection package files (XXX_C.ntf, XXX_M.ntf, XXX_R.ntf, or XXX_V.json) '
                'or a ChangeDetectionDetails instance. Got type {}'.format(change_details))
        self._change_details = change_details
        readers = (NITFReader(change_details.file_names['C']),
                   NITFReader(change_details.file_names['M']),
                   NITFReader(change_details.file_names['R']),)
        super(ChangeDetectionReader, self).__init__(readers)

    @property
    def features(self):
        # type: () -> Union[FeatureCollection, None]
        """
        None|FeatureCollection: The feature list, if it is defined.
        """

        return self._change_details.features


if __name__ == '__main__':
    logging.basicConfig(level='INFO')

    root_dir = os.path.expanduser('~/Downloads/AnamolousChangeDetection')
    this_file = os.path.join(root_dir, '16JUL24160156-P100_16JUL15160916-P100_20200618T140248_OBS_EOCD_C.ntf')

    details = ChangeDetectionDetails(this_file)
    reader = ChangeDetectionReader(details)

    from PIL import Image

    Image.fromarray(reader[:, :, 0]).save(os.path.join(root_dir, this_file[:-6] + '_C.jpeg'))
    Image.fromarray(reader[:, :, 1]).save(os.path.join(root_dir, this_file[:-6] + '_M.jpeg'))
    Image.fromarray(reader[:, :, 2]).save(os.path.join(root_dir, this_file[:-6] + '_R.jpeg'))
