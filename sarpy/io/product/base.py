"""
Base common features for product readers
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


from typing import Sequence, List, Tuple, Union, Optional

from sarpy.io.general.base import BaseReader
from sarpy.io.general.data_segment import DataSegment

from sarpy.io.product.sidd1_elements.SIDD import SIDDType as SIDDType1
from sarpy.io.product.sidd2_elements.SIDD import SIDDType as SIDDType2
from sarpy.io.complex.sicd_elements.SICD import SICDType


class SIDDTypeReader(BaseReader):
    def __init__(self,
                 data_segment: Union[None, DataSegment, Sequence[DataSegment]],
                 sidd_meta: Union[None, SIDDType2, SIDDType1, Sequence[SIDDType1], Sequence[SIDDType2]],
                 sicd_meta: Union[None, SICDType, Sequence[SICDType]],
                 close_segments: bool = True,
                 delete_files: Union[None, str, Sequence[str]] = None):
        """

        Parameters
        ----------
        data_segment : None|DataSegment|Sequence[DataSegment]
        sidd_meta : None|SIDDType1|SIDDType2|Sequence[SIDDType1]|Sequence[SIDDType2]
            The SIDD metadata object(s).
        sicd_meta : None|SICDType|Sequence[SICDType]
            The SICD metadata object(s).
        close_segments : bool
            Call segment.close() for each data segment on reader.close()?
        delete_files : None|Sequence[str]
            Any temp files which should be cleaned up on reader.close()?
            This will occur after closing segments.
        """

        if sidd_meta is None:
            self._sidd_meta = None
        elif isinstance(sidd_meta, (SIDDType1, SIDDType2)):
            self._sidd_meta = sidd_meta
        else:
            temp_list = []  # type: List[Union[SIDDType1]]
            for el in sidd_meta:
                if not isinstance(el, (SIDDType1, SIDDType2)):
                    raise TypeError(
                        'Got a collection for sidd_meta, and all elements are required '
                        'to be instances of SIDDType.')
                temp_list.append(el)
            self._sidd_meta = tuple(temp_list)

        if sicd_meta is None:
            self._sicd_meta = None
        elif isinstance(sicd_meta, SICDType):
            self._sicd_meta = (sicd_meta, )
        else:
            temp_list = []  # type: List[SICDType]
            for el in sicd_meta:
                if not isinstance(el, SICDType):
                    raise TypeError(
                        'Got a collection for sicd_meta, and all elements are required '
                        'to be instances of SICDType.')
                temp_list.append(el)
            self._sicd_meta = tuple(temp_list)

        BaseReader.__init__(
            self, data_segment, reader_type='SIDD', close_segments=close_segments, delete_files=delete_files)

    def _check_sizes(self) -> None:
        data_sizes = self.get_data_size_as_tuple()
        sidds = self.get_sidds_as_tuple()
        if len(data_sizes) != len(sidds):
            raise ValueError(
                'Got mismatched number of data segments ({}) and sidds ({})'.format(
                    len(data_sizes), len(sidds)))

        agree = True
        msg = ''
        for i, (data_size, sidd) in enumerate(zip(data_sizes, sidds)):
            expected_size = (sidd.Measurement.PixelFootprint.Row, sidd.Measurement.PixelFootprint.Col)
            if data_size[:2] != expected_size:
                agree = False
                msg += 'data segment at index {} has data size {}\n\t' \
                       'and expected size (from the sidd) {}\n'.format(i, data_size, expected_size)
        if not agree:
            raise ValueError(msg)

    @property
    def sidd_meta(self) -> Union[None, SIDDType1, SIDDType2, Tuple[SIDDType1, ...], Tuple[SIDDType2, ...]]:
        """
        None|SIDDType1|SIDDType2|Tuple[SIDDType1, ...]|Tuple[SIDDType2, ...]: the sidd meta_data collection.
        """

        return self._sidd_meta

    @property
    def sicd_meta(self) -> Optional[Tuple[SICDType, ...]]:
        """
        None|Tuple[SICDType, ...]: the sicd meta_data collection.
        """

        return self._sicd_meta

    def get_sidds_as_tuple(self) -> Union[None, Tuple[SIDDType1, ...], Tuple[SIDDType2, ...]]:
        """
        Get the sidd collection as a tuple - for simplicity and consistency of use.

        Returns
        -------
        None|Tuple[SIDDType1, ...]|Tuple[SIDDType2, ...]
        """

        if self.sidd_meta is None:
            return None
        elif isinstance(self._sidd_meta, tuple):
            return self.sidd_meta
        else:
            # noinspection PyRedundantParentheses
            return (self.sidd_meta, )
