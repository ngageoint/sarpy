"""
Base common features for complex readers
"""


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

from typing import Union, Tuple, Sequence, Callable
import numpy

from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.complex.sicd_elements.utils import is_general_match
from sarpy.io.general.base import BaseReader, FlatReader
from sarpy.io.general.data_segment import DataSegment, SubsetSegment
from sarpy.io.general.format_function import FormatFunction


class SICDTypeReader(BaseReader):
    """
    A class for ensuring common SICD reading functionality.

    **Changed in version 1.3.0** for reading changes.
    """

    def __init__(self,
                 data_segment: Union[None, DataSegment, Sequence[DataSegment]],
                 sicd_meta: Union[None, SICDType, Sequence[SICDType]],
                 close_segments: bool=True,
                 delete_files: Union[None, str, Sequence[str]]=None):
        """

        Parameters
        ----------
        data_segment : None|DataSegment|Sequence[DataSegment]
        sicd_meta : None|SICDType|Sequence[SICDType]
            The SICD metadata object(s).
        close_segments : bool
            Call segment.close() for each data segment on reader.close()?
        delete_files : None|Sequence[str]
            Any temp files which should be cleaned up on reader.close()?
            This will occur after closing segments.
        """

        if sicd_meta is None:
            self._sicd_meta = None
        elif isinstance(sicd_meta, SICDType):
            self._sicd_meta = sicd_meta
        else:
            temp_list = []
            for el in sicd_meta:
                if not isinstance(el, SICDType):
                    raise TypeError(
                        'Got a collection for sicd_meta, and all elements are required '
                        'to be instances of SICDType.')
                temp_list.append(el)
            self._sicd_meta = tuple(temp_list)

        BaseReader.__init__(
            self, data_segment, reader_type='SICD', close_segments=close_segments, delete_files=delete_files)

    def _check_sizes(self) -> None:
        data_sizes = self.get_data_size_as_tuple()
        sicds = self.get_sicds_as_tuple()
        agree = True
        msg = ''
        for i, (data_size, sicd) in enumerate(zip(data_sizes, sicds)):
            expected_size = (sicd.ImageData.NumRows, sicd.ImageData.NumCols)
            if data_size != expected_size:
                agree = False
                msg += 'data segment at index {} has data size {}\n\t' \
                       'and expected size (from the sicd) {}\n'.format(i, data_size, expected_size)
        if not agree:
            raise ValueError(msg)

    @property
    def sicd_meta(self) -> Union[None, SICDType, Tuple[SICDType, ...]]:
        """
        None|SICDType|Tuple[SICDType, ...]: the sicd meta_data or meta_data collection.
        """

        return self._sicd_meta

    def get_sicds_as_tuple(self) -> Union[None, Tuple[SICDType, ...]]:
        """
        Get the sicd or sicd collection as a tuple - for simplicity and consistency of use.

        Returns
        -------
        None|Tuple[SICDType, ...]
        """

        if self.sicd_meta is None:
            return None
        elif isinstance(self.sicd_meta, tuple):
            return self.sicd_meta
        else:
            # noinspection PyRedundantParentheses
            return (self.sicd_meta, )

    def get_sicd_partitions(self, match_function: Callable=is_general_match) -> Tuple[Tuple[int, ...], ...]:
        """
        Partition the sicd collection into sub-collections according to `match_function`,
        which is assumed to establish an equivalence relation.

        Parameters
        ----------
        match_function : callable
            This match function must have call signature `(SICDType, SICDType) -> bool`, and
            defaults to :func:`sarpy.io.complex.sicd_elements.utils.is_general_match`.
            This function is assumed reflexive, symmetric, and transitive.

        Returns
        -------
        Tuple[Tuple[int, ...], ...]
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

    def get_sicd_bands(self) -> Tuple[str, ...]:
        """
        Gets the list of bands for each sicd.

        Returns
        -------
        Tuple[str, ...]
        """

        return tuple(sicd.get_transmit_band_name() for sicd in self.get_sicds_as_tuple())

    def get_sicd_polarizations(self) -> Tuple[str, ...]:
        """
        Gets the list of polarizations for each sicd.

        Returns
        -------
        Tuple[str]
        """

        return tuple(sicd.get_processed_polarization() for sicd in self.get_sicds_as_tuple())


class FlatSICDReader(FlatReader, SICDTypeReader):
    """
    Create a sicd type reader directly from an array.

    **Changed in version 1.3.0** for reading changes.
    """

    def __init__(self,
                 sicd_meta,
                 underlying_array,
                 formatted_dtype: Union[None, str, numpy.dtype] = None,
                 formatted_shape: Union[None, Tuple[int, ...]] = None,
                 reverse_axes: Union[None, int, Sequence[int]] = None,
                 transpose_axes: Union[None, Tuple[int, ...]] = None,
                 format_function: Union[None, FormatFunction] = None,
                 close_segments: bool = True):
        """

        Parameters
        ----------
        sicd_meta : None|SICDType
            `None`, or the SICD metadata object
        underlying_array : numpy.ndarray
        formatted_dtype : None|str|numpy.dtype
        formatted_shape : None|Tuple[int, ...]
        reverse_axes : None|Sequence[int]
        transpose_axes : None|Tuple[int, ...]
        format_function : None|FormatFunction
        close_segments : bool
        """

        FlatReader.__init__(
            self, underlying_array,
            formatted_dtype=formatted_dtype, formatted_shape=formatted_shape,
            reverse_axes=reverse_axes, transpose_axes=transpose_axes,
            format_function=format_function, close_segments=close_segments)
        SICDTypeReader.__init__(self, None, sicd_meta)
        self._check_sizes()

    def write_to_file(self, output_file, check_older_version=False, check_existence=False):
        """
        Write a file for the given in-memory reader.

        Parameters
        ----------
        output_file : str
        check_older_version : bool
            Try to use a less recent version of SICD (1.1), for possible application compliance issues?
        check_existence : bool
            Should we check if the given file already exists, and raise an exception if so?
        """

        if not isinstance(output_file, str):
            raise TypeError(
                'output_file is expected to a be a string, got type {}'.format(type(output_file)))

        from sarpy.io.complex.sicd import SICDWriter
        with SICDWriter(
                output_file, self.sicd_meta,
                check_older_version=check_older_version, check_existence=check_existence) as writer:
            writer.write_chip(self[:, :], start_indices=(0, 0))


class SubsetSICDReader(SICDTypeReader):
    """
    Create a reader based on a specific subset of a given SICDTypeReader.

    **Changed in version 1.3.0** for reading changes.
    """

    def __init__(self, reader, row_bounds, column_bounds, index=0, close_parent=False):
        """

        Parameters
        ----------
        reader : SICDTypeReader
            The base reader.
        row_bounds : None|Tuple[int, int]
            Of the form `(min row, max row)`.
        column_bounds : None|Tuple[int, int]
            Of the form `(min column, max column)`.
        index : int
            The image index.
        close_parent : bool
        """

        sicd, row_bounds, column_bounds = reader.get_sicds_as_tuple()[index].create_subset_structure(
            row_bounds, column_bounds)

        parent_segment = reader.get_data_segment_as_tuple()[index]
        subset_definition = (slice(*row_bounds), slice(*column_bounds))
        data_segment = SubsetSegment(
            parent_segment, subset_definition, coordinate_basis='formatted', close_parent=close_parent)
        SICDTypeReader.__init__(self, data_segment, sicd)

    @property
    def file_name(self) -> None:
        return None
