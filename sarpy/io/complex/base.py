"""
Base common features for complex readers
"""


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

from typing import Union, Tuple
import numpy

from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.complex.sicd_elements.utils import is_general_match
from sarpy.io.general.base import FlatReader


class SICDTypeReader(object):
    """
    An abstract class for ensuring common SICD metadata functionality.

    This is intended to be used solely by extension.
    """

    def __init__(self, sicd_meta):
        """

        Parameters
        ----------
        sicd_meta : None|SICDType|Tuple[SICDType]
            `None`, the SICD metadata object, or tuple of objects
        """

        if isinstance(sicd_meta, list):
            sicd_meta = tuple(sicd_meta)

        # validate sicd_meta input
        if sicd_meta is None:
            pass
        elif isinstance(sicd_meta, tuple):
            for el in sicd_meta:
                if not isinstance(el, SICDType):
                    raise TypeError(
                        'Got a collection for sicd_meta, and all elements are required '
                        'to be instances of SICDType.')
        elif not isinstance(sicd_meta, SICDType):
            raise TypeError('sicd_meta argument is required to be a SICDType, or collection of SICDType objects')
        self._sicd_meta = sicd_meta

    @property
    def sicd_meta(self):
        # type: () -> Union[None, SICDType, Tuple[SICDType]]
        """
        None|SICDType|Tuple[SICDType]: the sicd meta_data or meta_data collection.
        """

        return self._sicd_meta

    def get_sicds_as_tuple(self):
        """
        Get the sicd or sicd collection as a tuple - for simplicity and consistency of use.

        Returns
        -------
        Tuple[SICDType]
        """

        if self._sicd_meta is None:
            return None
        elif isinstance(self._sicd_meta, tuple):
            return self._sicd_meta
        else:
            return (self._sicd_meta, )

    def get_sicd_partitions(self, match_function=is_general_match):
        """
        Partition the sicd collection into sub-collections according to `match_function`,
        which is assumed to establish an equivalence relation (reflexive, symmetric, and transitive).

        Parameters
        ----------
        match_function : callable
            This match function must have call signature `(SICDType, SICDType) -> bool`, and
            defaults to :func:`sarpy.io.complex.sicd_elements.utils.is_general_match`.
            This function is assumed reflexive, symmetric, and transitive.

        Returns
        -------
        Tuple[Tuple[int]]
        """

        sicds = self.get_sicds_as_tuple()
        # set up or state workspace
        count = len(sicds)
        matched = numpy.zeros((count,), dtype='bool')
        matches = []

        # assemble or match collections
        for i in range(count):
            if matched[i]:
                # it's already matched somewhere
                continue

            matched[i] = True  # shouldn't access backwards, but just to be thorough
            this_match = [i, ]
            for j in range(i + 1, count):
                if not matched[j] and match_function(sicds[i], sicds[j]):
                    matched[j] = True
                    this_match.append(j)
            matches.append(tuple(this_match))
        return tuple(matches)

    def get_sicd_bands(self):
        """
        Gets the list of bands for each sicd.

        Returns
        -------
        Tuple[str]
        """

        return tuple(sicd.get_transmit_band_name() for sicd in self.get_sicds_as_tuple())

    def get_sicd_polarizations(self):
        """
        Gets the list of polarizations for each sicd.

        Returns
        -------
        Tuple[str]
        """

        return tuple(sicd.get_processed_polarization() for sicd in self.get_sicds_as_tuple())


class FlatSICDReader(FlatReader, SICDTypeReader):
    def __init__(self, sicd_meta, array, output_bands=None, output_dtype=None,
                 symmetry=(False, False, False), transform_data=None, limit_to_raw_bands=None):
        """

        Parameters
        ----------
        sicd_meta : None|SICDType
            `None`, or the SICD metadata object
        array : numpy.ndarray
        output_bands : None|int
        output_dtype : None|str|numpy.dtype|numpy.number
        symmetry : tuple
            Describes any required data transformation. See the `symmetry` property.
        transform_data : None|str|Callable
            For data transformation after reading.
            If `None`, then no transformation will be applied. If `callable`, then
            this is expected to be the transformation method for the raw data. If
            string valued and `'complex'`, then the assumption is that real/imaginary
            components are stored in adjacent bands, which will be combined into a
            single band upon extraction. Other situations will yield and value error.
        limit_to_raw_bands : None|int|numpy.ndarray|list|tuple
            The collection of raw bands to which to read. `None` is all bands.
        """

        FlatReader.__init__(
            self, array, reader_type='SICD', output_bands=output_bands, output_dtype=output_dtype,
            symmetry=symmetry, transform_data=transform_data, limit_to_raw_bands=limit_to_raw_bands)
        SICDTypeReader.__init__(self, sicd_meta)
