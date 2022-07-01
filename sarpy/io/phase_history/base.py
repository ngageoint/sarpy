"""
Base structures for phase history readers and usage
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


from typing import Union, Tuple, Sequence, Dict, Optional

import numpy

from sarpy.io.general.base import BaseReader
from sarpy.io.general.data_segment import DataSegment
from sarpy.io.phase_history.cphd1_elements.CPHD import CPHDType as CPHDType1_0
from sarpy.io.phase_history.cphd0_3_elements.CPHD import CPHDType as CPHDType0_3


class CPHDTypeReader(BaseReader):
    """
    A class for common CPHD reading functionality.

    **Updated in version 1.3.0**
    """

    def __init__(
            self,
            data_segment: Union[None, DataSegment, Sequence[DataSegment]],
            cphd_meta: Union[None, CPHDType1_0, CPHDType0_3],
            close_segments: bool=True,
            delete_files: Union[None, str, Sequence[str]]=None):
        """

        Parameters
        ----------
        data_segment : None|DataSegment|Sequence[DataSegment]
        cphd_meta : None|CPHDType1_0|CPHDType0_3
            The CPHD metadata object
        close_segments : bool
            Call segment.close() for each data segment on reader.close()?
        delete_files : None|Sequence[str]
            Any temp files which should be cleaned up on reader.close()?
            This will occur after closing segments.
        """

        if cphd_meta is None:
            self._cphd_meta = None
        elif isinstance(cphd_meta, (CPHDType1_0, CPHDType0_3)):
            self._cphd_meta = cphd_meta
        else:
            raise TypeError(
                'The cphd_meta must be of type CPHDType, got `{}`'.format(type(cphd_meta)))

        BaseReader.__init__(
            self, data_segment, reader_type='CPHD', close_segments=close_segments, delete_files=delete_files)

    @property
    def cphd_meta(self) -> Union[None, CPHDType1_0, CPHDType0_3]:
        """
        None|CPHDType1_0|CPHDType0_3: the cphd meta_data.
        """

        return self._cphd_meta

    def read_support_array(
            self,
            index: Union[int, str],
            *ranges: Sequence[Union[None, int, Tuple[int, ...], slice]]) -> numpy.ndarray:
        """
        Read the support array.

        Parameters
        ----------
        index : int|str
            The support array integer index.
        ranges : Sequence[None|int|Tuple[int, ...]|slice]
            The slice definition appropriate for support array usage.

        Returns
        -------
        numpy.ndarray

        Raises
        ------
        TypeError
            If called on a reader which doesn't support this.
        """

        raise TypeError('Class {} does not provide support arrays'.format(type(self)))

    def read_support_block(self) -> Dict[str, numpy.ndarray]:
        """
        Reads the entirety of support block(s).

        Returns
        -------
        Dict[str, numpy.ndarray]
            Dictionary of `numpy.ndarray` containing the support arrays.
        """

        raise TypeError('Class {} does not provide support arrays'.format(type(self)))

    def read_pvp_variable(
            self,
            variable: str,
            index: Union[int, str],
            the_range: Union[None, int, Tuple[int, ...], slice]=None) -> Optional[numpy.ndarray]:
        """
        Read the vector parameter for the given `variable` and CPHD channel.

        Parameters
        ----------
        variable : str
        index : int|str
            The channel index or identifier.
        the_range : None|int|Tuple[int, ...]|slice
            The indices for the vector parameter. `None` returns all,
            a integer returns the single value at that location, otherwise
            the input determines a slice.

        Returns
        -------
        None|numpy.ndarray
            This will return None if there is no such variable, otherwise the data.
        """

        raise NotImplementedError

    def read_pvp_array(
            self,
            index: Union[int, str],
            the_range: Union[None, int, Tuple[int, ...], slice]=None) -> numpy.ndarray:
        """
        Read the PVP array from the requested channel.

        Parameters
        ----------
        index : int|str
            The support array integer index (of cphd.Data.Channels list) or identifier.
        the_range : None|int|Tuple[int, ...]|slice
            The indices for the vector parameter. `None` returns all,
            a integer returns the single value at that location, otherwise
            the input determines a slice.

        Returns
        -------
        pvp_array : numpy.ndarray
        """

        raise NotImplementedError

    def read_pvp_block(self) -> Dict[Union[int, str], numpy.ndarray]:
        """
        Reads the entirety of the PVP block(s).

        Returns
        -------
        Dict[Union[int, str], numpy.ndarray]
            Dictionary containing the PVP arrays.
        """

        raise NotImplementedError

    def read_signal_block(self) -> Dict[Union[int, str], numpy.ndarray]:
        """
        Reads the entirety of signal block(s), with data formatted as complex64
        (after accounting for AmpSF).

        Returns
        -------
        Dict[Union[int, str], numpy.ndarray]
            Dictionary of `numpy.ndarray` containing the signal arrays.
        """

        raise NotImplementedError

    def read_signal_block_raw(self) -> Dict[Union[int, str], numpy.ndarray]:
        """
        Reads the entirety of signal block(s), with data formatted in file
        storage format (no converting to complex, no consideration of AmpSF).

        Returns
        -------
        Dict[Union[int, str], numpy.ndarray]
            Dictionary of `numpy.ndarray` containing the signal arrays.
        """

        raise NotImplementedError
