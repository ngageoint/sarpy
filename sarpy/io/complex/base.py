"""
Base common features for complex readers
"""


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

import os
from typing import Union, Tuple, BinaryIO
import numpy
import warnings

from sarpy.compliance import string_types
from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.complex.sicd_elements.utils import is_general_match
from sarpy.io.general.base import FlatReader, BaseChipper, is_file_like

try:
    import h5py
except ImportError:
    h5py = None


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


class H5Chipper(BaseChipper):
    __slots__ = ('_file_name', '_band_name')

    def __init__(self, file_name, band_name, data_size, symmetry, transform_data='COMPLEX'):
        if h5py is None:
            raise ImportError("Can't read hdf5 files, because the h5py dependency is missing.")
        self._file_name = file_name
        self._band_name = band_name
        super(H5Chipper, self).__init__(data_size, symmetry=symmetry, transform_data=transform_data)

    def _read_raw_fun(self, range1, range2):
        def reorder(tr):
            if tr[2] > 0:
                return tr, False
            else:
                if tr[1] == -1 and tr[2] < 0:
                    return (0, tr[0]+1, -tr[2]), True
                else:
                    return (tr[1], tr[0], -tr[2]), True

        r1, r2 = self._reorder_arguments(range1, range2)
        r1, rev1 = reorder(r1)
        r2, rev2 = reorder(r2)
        with h5py.File(self._file_name, 'r') as hf:
            gp = hf[self._band_name]
            if not isinstance(gp, h5py.Dataset):
                raise ValueError(
                    'hdf5 group {} is expected to be a dataset, got type {}'.format(self._band_name, type(gp)))
            if len(gp.shape) not in (2, 3):
                raise ValueError('Dataset {} has unexpected shape {}'.format(self._band_name, gp.shape))

            if len(gp.shape) == 3:
                data = gp[r1[0]:r1[1]:r1[2], r2[0]:r2[1]:r2[2], :]
            else:
                data = gp[r1[0]:r1[1]:r1[2], r2[0]:r2[1]:r2[2]]

        if rev1 and rev2:
            return data[::-1, ::-1]
        elif rev1:
            return data[::-1, :]
        elif rev2:
            return data[:, ::-1]
        else:
            return data


def is_hdf5(file_name):
    """
    Test whether the given input file is an hdf5 file.

    Parameters
    ----------
    file_name : str|BinaryIO

    Returns
    -------
    bool
    """

    if is_file_like(file_name):
        current_location = file_name.tell()
        file_name.seek(0, os.SEEK_SET)
        header = file_name.read(4)
        file_name.seek(current_location, os.SEEK_SET)
    elif isinstance(file_name, string_types):
        if not os.path.isfile(file_name):
            return False

        with open(file_name, 'rb') as fi:
            header = fi.read(4)
    else:
        return False

    out = (header == b'\x89HDF')
    if out and h5py is None:
        warnings.warn('The h5py library was not successfully imported, and no hdf5 files can be read')
    return out
