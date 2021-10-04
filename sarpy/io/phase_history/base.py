"""
Base structures for phase history readers and usage
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

from typing import Union, Tuple

import numpy

from sarpy.io.general.base import AbstractReader
from sarpy.io.phase_history.cphd1_elements.CPHD import CPHDType as CPHDType1_0
from sarpy.io.phase_history.cphd0_3_elements.CPHD import CPHDType as CPHDType0_3


class CPHDTypeReader(AbstractReader):
    """
    An abstract class for ensuring common CPHD functionality.

    This is intended to be used solely in conjunction with implementing a
    legitimate reader.
    """

    def __init__(self, cphd_meta):
        """

        Parameters
        ----------
        cphd_meta : None|CPHDType1_0|CPHDType0_3
            `None`, the CPHD metadata object
        """

        if cphd_meta is None:
            self._cphd_meta = None
        elif isinstance(cphd_meta, (CPHDType1_0, CPHDType0_3)):
            self._cphd_meta = cphd_meta
        else:
            raise TypeError(
                'The cphd_meta must be of type CPHDType, got `{}`'.format(type(cphd_meta)))

    @property
    def cphd_meta(self):
        # type: () -> Union[None, CPHDType1_0, CPHDType0_3]
        """
        None|CPHDType1_0|CPHDType0_3: the cphd meta_data.
        """

        return self._cphd_meta

    def read_support_array(self, index, dim1_range, dim2_range):
        # type: (Union[int, str], Union[None, int, Tuple[int, int], Tuple[int, int, int]], Union[None, int, Tuple[int, int], Tuple[int, int, int]]) -> numpy.ndarray
        """
        Read the support array.

        Parameters
        ----------
        index : int|str
            The support array integer index.
        dim1_range : None|int|Tuple[int, int]|Tuple[int, int, int]
            The row data selection of the form `[start, [stop, [stride]]]`, and
            `None` defaults to all rows (i.e. `(0, NumRows, 1)`)
        dim2_range : None|int|Tuple[int, int]|Tuple[int, int, int]
            The column data selection of the form `[start, [stop, [stride]]]`, and
            `None` defaults to all rows (i.e. `(0, NumCols, 1)`)

        Returns
        -------
        numpy.ndarray

        Raises
        ------
        TypeError
            If called on a reader which doesn't support this.
        """

        raise TypeError('Class {} does not provide support arrays'.format(type(self)))

    def read_support_block(self):
        """
        Reads the entirety of support block(s).

        Returns
        -------
        Dict[str, numpy.ndarray]
            Dictionary of `numpy.ndarray` containing the support arrays.
        """

        raise TypeError('Class {} does not provide support arrays'.format(type(self)))

    def read_pvp_variable(self, variable, index, the_range=None):
        """
        Read the vector parameter for the given `variable` and CPHD channel.

        Parameters
        ----------
        variable : str
        index : int|str
            The channel index or identifier.
        the_range : None|int|List[int]|Tuple[int]
            The indices for the vector parameter. `None` returns all, otherwise
            a slice in the (non-traditional) form `([start, [stop, [stride]]])`.

        Returns
        -------
        None|numpy.ndarray
            This will return None if there is no such variable, otherwise the data.
        """

        raise NotImplementedError

    def read_pvp_array(self, index, the_range=None):
        """
        Read the PVP array from the requested channel.

        Parameters
        ----------
        index : int|str
            The support array integer index (of cphd.Data.Channels list) or identifier.
        the_range : None|int|List[int]|Tuple[int]
            The indices for the vector parameter. `None` returns all, otherwise
            a slice in the (non-traditional) form `([start, [stop, [stride]]])`.

        Returns
        -------
        pvp_array : numpy.ndarray
        """

        raise NotImplementedError

    def read_pvp_block(self):
        """
        Reads the entirety of the PVP block(s).

        Returns
        -------
        Dict[Union[int, str], numpy.ndarray]
            Dictionary of `numpy.ndarray` containing the PVP arrays.
        """

        raise NotImplementedError

    def read_signal_block(self):
        """
        Reads the entirety of signal block(s).

        Returns
        -------
        Dict[Union[int, str], numpy.ndarray]
            Dictionary of `numpy.ndarray` containing the support arrays.
        """

        raise NotImplementedError
