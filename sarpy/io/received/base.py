"""
Base structures for received signal data readers and usage
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

from typing import Union, Tuple, Sequence, Dict, Optional

import numpy

from sarpy.io.general.base import BaseReader
from sarpy.io.general.data_segment import DataSegment
from sarpy.io.received.crsd1_elements.CRSD import CRSDType as CRSDType1_0


class CRSDTypeReader(BaseReader):
    """
    A class for ensuring common CRSD reading functionality.

    **Updated in version 1.3.0** for reading changes.
    """

    def __init__(self,
                 data_segment: Union[None, DataSegment, Sequence[DataSegment]],
                 crsd_meta: Union[None, CRSDType1_0],
                 close_segments: bool=True,
                 delete_files: Union[None, str, Sequence[str]]=None):
        """

        Parameters
        ----------
        data_segment : None|DataSegment|Sequence[DataSegment]
        crsd_meta : None|CRSDType1_0
            The CRSD metadata object
        close_segments : bool
            Call segment.close() for each data segment on reader.close()?
        delete_files : None|Sequence[str]
            Any temp files which should be cleaned up on reader.close()?
            This will occur after closing segments.
        """

        if crsd_meta is None:
            self._crsd_meta = None
        elif isinstance(crsd_meta, CRSDType1_0):
            self._crsd_meta = crsd_meta
        else:
            raise TypeError(
                'The crsd_meta must be of type CRSDType, got `{}`'.format(type(crsd_meta)))

        BaseReader.__init__(
            self, data_segment, reader_type='CRSD', close_segments=close_segments, delete_files=delete_files)

    @property
    def crsd_meta(self) -> Union[None, CRSDType1_0]:
        """
        None|CRSDType1_0: the crsd meta_data.
        """

        return self._crsd_meta

    def read_support_array(self,
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

        raise NotImplementedError

    def read_support_block(self) -> Dict[str, numpy.ndarray]:
        """
        Reads the entirety of support block(s).

        Returns
        -------
        Dict[str, numpy.ndarray]
            Dictionary of `numpy.ndarray` containing the support arrays.
        """

        raise NotImplementedError

    def read_pvp_variable(
            self,
            variable: str,
            index: Union[int, str],
            the_range: Union[None, int, Tuple[int, ...], slice]=None) -> Optional[numpy.ndarray]:
        """
        Read the vector parameter for the given `variable` and CRSD channel.

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

    def read_pvp_block(self) -> Dict[str, numpy.ndarray]:
        """
        Reads the entirety of the PVP block(s).

        Returns
        -------
        Dict[str, numpy.ndarray]
            Dictionary containing the PVP arrays.
        """

        raise NotImplementedError

    def read_signal_block(self) -> Dict[str, numpy.ndarray]:
        """
        Reads the entirety of signal block(s), with data formatted as complex64
        (after accounting for AmpSF).

        Returns
        -------
        Dict[str, numpy.ndarray]
            Dictionary of `numpy.ndarray` containing the signal arrays.
        """

        raise NotImplementedError

    def read_signal_block_raw(self) -> Dict[str, numpy.ndarray]:
        """
        Reads the entirety of signal block(s), with data formatted in file
        storage format (no converting to complex, no consideration of AmpSF).

        Returns
        -------
        Dict[str, numpy.ndarray]
            Dictionary of `numpy.ndarray` containing the signal arrays.
        """

        raise NotImplementedError
