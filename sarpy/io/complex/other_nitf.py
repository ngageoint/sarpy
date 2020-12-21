# -*- coding: utf-8 -*-
"""
Work in progress for reading some other kind of complex NITF.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

from typing import Union

import numpy

from sarpy.compliance import string_types

from sarpy.io.general.nitf import NITFDetails, NITFReader
from sarpy.io.general.nitf_elements.image import ImageSegmentHeader, ImageSegmentHeader0
from sarpy.io.complex.utils import extract_sicd


# NB: DO NOT implement is_a() here. This should definitely happen after other
# readers

def final_attempt(file_name):
    """
    Contingency check to open for some other complex NITF type file.
    Returns a reader instance, if so.

    Parameters
    ----------
    file_name : str
        the file_name to check

    Returns
    -------
    ComplexNITFReader|None
    """

    try:
        nitf_details = ComplexNITFDetails(file_name)
        print('File {} is determined to be some other format complex NITF.')
        return ComplexNITFReader(nitf_details)
    except IOError:
        return None


class ComplexNITFDetails(NITFDetails):
    """
    Details object for NITF file containing complex data.
    """

    __slots__ = ('_complex_segments', '_sicd_meta', '_symmetry', '_split_bands')

    def __init__(self, file_name, symmetry=(False, False, False), split_bands=True):
        """

        Parameters
        ----------
        file_name : str
            file name for a NITF file containing a complex SICD
        symmetry : tuple
        split_bands : bool
            Split multiple complex bands into single bands?
        """

        self._split_bands = split_bands
        self._symmetry = symmetry
        self._sicd_meta = None
        self._complex_segments = None
        super(ComplexNITFDetails, self).__init__(file_name)
        if self._nitf_header.ImageSegments.subhead_sizes.size == 0:
            raise IOError('There are no image segments defined.')
        self._find_complex_image_segments()
        if self._complex_segments is None:
            raise IOError('No complex valued (I/Q) image segments found in file {}'.format(file_name))

    @property
    def complex_segments(self):
        """
        List[dict]: The image details for each relevant image segment.
        """

        return self._complex_segments

    @property
    def sicd_meta(self):
        """
        Tuple[SICDType]: The best inferred sicd structures.
        """

        return self._sicd_meta

    def _find_complex_image_segments(self):
        """
        Find complex image segments.

        Returns
        -------
        None
        """

        def extract_band_details(image_header):
            # type: (Union[ImageSegmentHeader, ImageSegmentHeader0]) -> int
            bands = len(image_header.Bands)
            if image_header.ICAT.strip() in ['SAR', 'SARIQ'] and ((bands % 2) == 0):
                # TODO: account for PVType == 'C' and ISUBCAT = 'M'/'P'
                cont = True
                for j in range(0, bands, 2):
                    cont &= (image_header.Bands[j].ISUBCAT == 'I'
                             and image_header.Bands[j+1].ISUBCAT == 'Q')
                return bands
            return 0

        sicd_meta = []
        complex_segments = []
        for i, img_header in enumerate(self.img_headers):
            complex_bands = extract_band_details(img_header)
            if complex_bands > 0:
                sicd = extract_sicd(img_header, self._symmetry)
                if self._split_bands and complex_bands > 2:
                    for j in range(0, complex_bands, 2):
                        complex_segments.append(
                            {'index': i, 'output_bands': 1, 'limit_to_raw_bands': numpy.array([j, j+1], dtype='int32')})
                        sicd_meta.append(sicd.copy())
                else:
                    complex_segments.append({'index': i, })
                    sicd_meta.append(sicd)

        if len(sicd_meta) > 0:
            self._complex_segments = complex_segments
            self._sicd_meta = tuple(sicd_meta)


class ComplexNITFReader(NITFReader):
    """
    A reader for complex valued NITF elements, this should be explicitly tried AFTER
    the SICDReader.
    """

    def __init__(self, nitf_details, symmetry=(False, False, False), split_bands=True):
        """

        Parameters
        ----------
        nitf_details : str|ComplexNITFDetails
        symmetry : tuple
            Passed through to ComplexNITFDetails() in the event that `nitf_details` is a file name.
        split_bands : bool
            Passed through to ComplexNITFDetails() in the event that `nitf_details` is a file name.
        """

        if isinstance(nitf_details, string_types):
            nitf_details = ComplexNITFDetails(nitf_details, symmetry=symmetry, split_bands=split_bands)
        if not isinstance(nitf_details, ComplexNITFDetails):
            raise TypeError('The input argument for ComplexNITFReader must be a filename or '
                            'ComplexNITFDetails object.')
        super(ComplexNITFReader, self).__init__(nitf_details, reader_type="SICD", symmetry=symmetry)

    @property
    def nitf_details(self):
        # type: () -> ComplexNITFDetails
        """
        ComplexNITFDetails: The NITF details object.
        """

        # noinspection PyTypeChecker
        return self._nitf_details

    def _find_segments(self):
        return [[entry['index'], ] for entry in self.nitf_details.complex_segments]

    def _construct_chipper(self, segment, index):
        entry = self.nitf_details.complex_segments[index]
        if entry['index'] != segment[0]:
            raise ValueError('Got incompatible entries.')
        return self._define_chipper(
            entry['index'], output_bands=entry.get('output_bands', None),
            limit_to_raw_bands=entry.get('limit_to_raw_bands', None))
