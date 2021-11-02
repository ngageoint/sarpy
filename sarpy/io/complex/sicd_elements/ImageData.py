"""
The ImageData definition.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


from typing import List, Union

import numpy

from sarpy.io.xml.base import Serializable, Arrayable, SerializableArray
from sarpy.io.xml.descriptors import IntegerDescriptor, FloatArrayDescriptor, \
    StringEnumDescriptor, SerializableDescriptor, SerializableArrayDescriptor
from sarpy.geometry.geometry_elements import LinearRing

from .base import DEFAULT_STRICT, FLOAT_FORMAT
from .blocks import RowColType, RowColArrayElement


class FullImageType(Serializable, Arrayable):
    """
    The full image product attributes.
    """

    _fields = ('NumRows', 'NumCols')
    _required = _fields
    # descriptors
    NumRows = IntegerDescriptor(
        'NumRows', _required, strict=True,
        docstring='Number of rows in the original full image product. May include zero pixels.')  # type: int
    NumCols = IntegerDescriptor(
        'NumCols', _required, strict=True,
        docstring='Number of columns in the original full image product. May include zero pixels.')  # type: int

    def __init__(self, NumRows=None, NumCols=None, **kwargs):
        """

        Parameters
        ----------
        NumRows : int
        NumCols : int
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.NumRows, self.NumCols = NumRows, NumCols
        super(FullImageType, self).__init__(**kwargs)

    def get_array(self, dtype=numpy.int64):
        """Gets an array representation of the class instance.

        Parameters
        ----------
        dtype : str|numpy.dtype|numpy.number
            numpy data type of the return

        Returns
        -------
        numpy.ndarray
            array of the form `[X,Y,Z]`
        """

        return numpy.array([self.NumRows, self.NumCols], dtype=dtype)

    @classmethod
    def from_array(cls, array):
        """
        Create from an array type entry.

        Parameters
        ----------
        array: numpy.ndarray|list|tuple
            assumed `[NumRows, NumCols]`

        Returns
        -------
        FullImageType
        """

        if isinstance(array, (numpy.ndarray, list, tuple)):
            if len(array) < 2:
                raise ValueError('Expected array to be of length 2, and received {}'.format(array))
            return cls(NumRows=array[0], NumCols=array[1])
        raise ValueError('Expected array to be numpy.ndarray, list, or tuple, got {}'.format(type(array)))


class ImageDataType(Serializable):
    """The image pixel data."""
    _collections_tags = {
        'AmpTable': {'array': True, 'child_tag': 'Amplitude'},
        'ValidData': {'array': True, 'child_tag': 'Vertex'},
    }
    _fields = (
        'PixelType', 'AmpTable', 'NumRows', 'NumCols', 'FirstRow', 'FirstCol', 'FullImage', 'SCPPixel', 'ValidData')
    _required = ('PixelType', 'NumRows', 'NumCols', 'FirstRow', 'FirstCol', 'FullImage', 'SCPPixel')
    _numeric_format = {'AmpTable': FLOAT_FORMAT}
    _PIXEL_TYPE_VALUES = ("RE32F_IM32F", "RE16I_IM16I", "AMP8I_PHS8I")
    # descriptors
    PixelType = StringEnumDescriptor(
        'PixelType', _PIXEL_TYPE_VALUES, _required, strict=True,
        docstring="The PixelType attribute which specifies the interpretation of the file data.")  # type: str
    AmpTable = FloatArrayDescriptor(
        'AmpTable', _collections_tags, _required, strict=DEFAULT_STRICT,
        minimum_length=256, maximum_length=256,
        docstring="The amplitude look-up table. This is required if "
                  "`PixelType == 'AMP8I_PHS8I'`")  # type: numpy.ndarray
    NumRows = IntegerDescriptor(
        'NumRows', _required, strict=True,
        docstring='The number of Rows in the product. May include zero rows.')  # type: int
    NumCols = IntegerDescriptor(
        'NumCols', _required, strict=True,
        docstring='The number of Columns in the product. May include zero rows.')  # type: int
    FirstRow = IntegerDescriptor(
        'FirstRow', _required, strict=DEFAULT_STRICT,
        docstring='Global row index of the first row in the product. '
                  'Equal to 0 in full image product.')  # type: int
    FirstCol = IntegerDescriptor(
        'FirstCol', _required, strict=DEFAULT_STRICT,
        docstring='Global column index of the first column in the product. '
                  'Equal to 0 in full image product.')  # type: int
    FullImage = SerializableDescriptor(
        'FullImage', FullImageType, _required, strict=DEFAULT_STRICT,
        docstring='Original full image product.')  # type: FullImageType
    SCPPixel = SerializableDescriptor(
        'SCPPixel', RowColType, _required, strict=DEFAULT_STRICT,
        docstring='Scene Center Point pixel global row and column index. Should be located near the '
                  'center of the full image.')  # type: RowColType
    ValidData = SerializableArrayDescriptor(
        'ValidData', RowColArrayElement, _collections_tags, _required, strict=DEFAULT_STRICT, minimum_length=3,
        docstring='Indicates the full image includes both valid data and some zero filled pixels. '
                  'Simple polygon encloses the valid data (may include some zero filled pixels for simplification). '
                  'Vertices in clockwise order.')  # type: Union[SerializableArray, List[RowColArrayElement]]

    def __init__(self, PixelType=None, AmpTable=None, NumRows=None, NumCols=None,
                 FirstRow=None, FirstCol=None, FullImage=None, SCPPixel=None, ValidData=None, **kwargs):
        """

        Parameters
        ----------
        PixelType : str
        AmpTable : numpy.ndarray|list|tuple
        NumRows : int
        NumCols : int
        FirstRow : int
        FirstCol : int
        FullImage : FullImageType|numpy.ndarray|list|tuple
        SCPPixel : RowColType|numpy.ndarray|list|tuple
        ValidData : SerializableArray|List[RowColArrayElement]|numpy.ndarray|list|tuple
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.PixelType = PixelType
        self.AmpTable = AmpTable
        self.NumRows, self.NumCols = NumRows, NumCols
        self.FirstRow, self.FirstCol = FirstRow, FirstCol
        self.FullImage = FullImage
        self.SCPPixel = SCPPixel
        self.ValidData = ValidData
        super(ImageDataType, self).__init__(**kwargs)

    def _check_valid_data(self):
        if self.ValidData is None:
            return True
        if len(self.ValidData) < 2:
            return True

        value = True
        valid_data = self.ValidData.get_array(dtype='float64')
        lin_ring = LinearRing(coordinates=valid_data)
        area = lin_ring.get_area()
        if area == 0:
            self.log_validity_error('ValidData encloses no area.')
            value = False
        elif area > 0:
            self.log_validity_error(
                "ValidData must be traversed in clockwise direction.")
            value = False
        for i, entry in enumerate(valid_data):
            if not ((self.FirstRow <= entry[0] <= self.FirstRow + self.NumRows) and
                    (self.FirstCol <= entry[1] <= self.FirstCol + self.NumCols)):
                self.log_validity_warning(
                    'ValidData entry {} is not contained in the image bounds'.format(i))
                value = False
        return value

    def _basic_validity_check(self):
        condition = super(ImageDataType, self)._basic_validity_check()
        if (self.PixelType == 'AMP8I_PHS8I') and (self.AmpTable is None):
            self.log_validity_error("We have `PixelType='AMP8I_PHS8I'` and `AmpTable` is not defined for ImageDataType.")
            condition = False
        if (self.PixelType != 'AMP8I_PHS8I') and (self.AmpTable is not None):
            self.log_validity_error("We have `PixelType != 'AMP8I_PHS8I'` and `AmpTable` is defined for ImageDataType.")
            condition = False
        if (self.ValidData is not None) and (len(self.ValidData) < 3):
            self.log_validity_error("We have `ValidData` defined with fewer than 3 entries.")
            condition = False
        condition &= self._check_valid_data()
        return condition

    def get_valid_vertex_data(self, dtype=numpy.int64):
        """
        Gets an array of `[row, col]` indices defining the valid data. If this is not viable, then `None`
        will be returned.

        Parameters
        ----------
        dtype : object
            the data type for the array

        Returns
        -------
        numpy.ndarray|None
        """

        if self.ValidData is None:
            return None
        out = numpy.zeros((self.ValidData.size, 2), dtype=dtype)
        for i, entry in enumerate(self.ValidData):
            out[i, :] = entry.get_array(dtype=dtype)
        return out

    def get_full_vertex_data(self, dtype=numpy.int64):
        """
        Gets an array of `[row, col]` indices defining the full vertex data. If this is not viable, then `None`
        will be returned.

        Parameters
        ----------
        dtype : object
            the data type for the array

        Returns
        -------
        numpy.ndarray|None
        """

        if self.NumRows is None or self.NumCols is None:
            return None
        return numpy.array(
            [[0, 0], [0, self.NumCols - 1], [self.NumRows - 1, self.NumCols - 1], [self.NumRows - 1, 0]], dtype=dtype)
