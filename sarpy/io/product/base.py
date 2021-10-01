"""
Base common features for product readers
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


from typing import Sequence, List, Tuple, Union

from sarpy.io.general.base import AbstractReader
from sarpy.io.product.sidd1_elements.SIDD import SIDDType as SIDDType1
from sarpy.io.product.sidd2_elements.SIDD import SIDDType as SIDDType2
from sarpy.io.complex.sicd_elements.SICD import SICDType


class SIDDTypeReader(AbstractReader):

    def __init__(self, sidd_meta, sicd_meta):
        """

        Parameters
        ----------
        sidd_meta : None|SIDDType1|SIDDType2|Sequence[SIDDType1]|Sequence[SIDDType2]
            The SIDD metadata object(s), if provided
        sicd_meta : None|SICDType|Sequence[SICDType]
            the SICD metadata object(s), if provided
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
            temp_list = [] # type: List[SICDType]
            for el in sicd_meta:
                if not isinstance(el, SICDType):
                    raise TypeError(
                        'Got a collection for sicd_meta, and all elements are required '
                        'to be instances of SICDType.')
                temp_list.append(el)
            self._sicd_meta = tuple(temp_list)

    @property
    def sidd_meta(self):
        # type: () -> Union[None, SIDDType1, SIDDType2, Tuple[SIDDType1], Tuple[SIDDType2]]
        """
        None|SIDDType1|SIDDType2|Tuple[SIDDType1]|Tuple[SIDDType2]: the sidd meta_data collection.
        """

        return self._sidd_meta

    @property
    def sicd_meta(self):
        # type: () -> Union[None, Tuple[SICDType]]
        """
        None|Tuple[SICDType]: the sicd meta_data collection.
        """

        return self._sicd_meta

    def get_sidds_as_tuple(self):
        """
        Get the sidd collection as a tuple - for simplicity and consistency of use.

        Returns
        -------
        Tuple[SIDDType1]|Tuple[SIDDType2]
        """

        if self._sidd_meta is None:
            return None
        elif isinstance(self._sidd_meta, tuple):
            return self._sidd_meta
        else:
            return (self._sidd_meta, )
