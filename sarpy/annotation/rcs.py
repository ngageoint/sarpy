"""
This module provides structures for annotating a given SICD type file for RCS
calculations
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


import logging
from collections import OrderedDict, defaultdict
import json
from typing import Union, Any, List

import numpy

from sarpy.geometry.geometry_elements import Jsonable, Polygon, MultiPolygon
from sarpy.annotation.base import AnnotationFeature, AnnotationProperties, \
    AnnotationCollection, FileAnnotationCollection

from sarpy.io.complex.base import SICDTypeReader
from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.complex.utils import get_im_physical_coords


_RCS_VERSION = "RCS:1.0"
logger = logging.getLogger(__name__)

DEFAULT_NAME_MAPPING = OrderedDict(
    RCS='RCSSFPoly',
    BetaZero='BetaZeroSFPoly',
    GammaZero='GammaZeroSFPoly',
    SigmaZero='SigmaZeroSFPoly')


def _get_polygon_bounds(polygon, data_size):
    """
    Gets the row/column bounds for the polygon and a polygon inclusion mask for
    the defined rectangular pixel grid.

    Parameters
    ----------
    polygon : Polygon
    data_size : Tuple[int, int]

    Returns
    -------
    row_bounds : Tuple[int, int]
        The lower and upper bounds for the rows.
    col_bounds : Tuple[int, int]
        The lower and upper bounds for the columns.
    mask: numpy.ndarray
        The boolean inclusion mask.
    """

    if not isinstance(polygon, Polygon):
        raise TypeError('polygon must be an instance of Polygon, got type {}'.format(type(polygon)))

    bounding_box = polygon.get_bbox()
    if len(bounding_box) != 4:
        raise ValueError('Got unexpected bounding box {}'.format(bounding_box))

    row_min = max(0, int(numpy.floor(bounding_box[0])))
    row_max = min(int(numpy.floor(bounding_box[2])) + 1, data_size[0])

    col_min = max(0, int(numpy.floor(bounding_box[1])))
    col_max = min(int(numpy.floor(bounding_box[3])) + 1, data_size[1])

    row_bounds = (row_min, row_max)
    col_bounds = (col_min, col_max)
    mask = polygon.grid_contained(
        numpy.arange(row_bounds[0], row_bounds[1]),
        numpy.arange(col_bounds[0], col_bounds[1]))
    return row_bounds, col_bounds, mask


def create_rcs_value_collection_for_reader(reader, polygon):
    """
    Given a SICD type reader and a polygon with coordinates in pixel space
    (all sicd footprint assumed applicable), construct the `RCSValueCollection`.

    Parameters
    ----------
    reader : SICDTypeReader
    polygon : Polygon|MultiPolygon

    Returns
    -------
    RCSValueCollection
    """

    def evaluate_sicd(the_sicd):
        # type: (SICDType) -> (bool, bool)
        if the_sicd.Radiometric is None:
            return False, False

        if the_sicd.Radiometric.NoiseLevel is None:
            return True, False
        if the_sicd.Radiometric.NoiseLevel.NoiseLevelType == 'ABSOLUTE':
            return True, True
        else:
            return True, False

    def get_stat_entries():
        def get_empty_dict():
            return dict(total=0.0, total2=0.0, count=0, min=numpy.inf, max=-numpy.inf)

        return defaultdict(
            lambda: dict(value=get_empty_dict(), noise=get_empty_dict()) if has_noise else dict(value=get_empty_dict()))

    def calculate_statistics(array, the_entry):
        # type: (numpy.ndarray, dict) -> None
        the_entry['total'] += numpy.sum(array)
        the_entry['total2'] += numpy.sum(array*array)
        the_entry['count'] += array.size
        the_entry['min'] = min(the_entry['min'], numpy.min(array))
        the_entry['max'] = max(the_entry['max'], numpy.max(array))

    def get_total_rcs(the_stats, the_pol, the_ind, oversamp):
        if has_radiometric:
            the_entry = the_stats['RCS']
            name = 'TotalRCS'
            val = the_entry['value']['total']/oversamp
            noise_val = the_entry['noise']['total']/oversamp if has_noise else None
        else:
            the_entry = the_stats['PixelPower']
            name = 'TotalPixelPower'
            val = the_entry['value']['total']
            noise_val = the_entry['noise']['total'] if has_noise else None

        out = RCSValue(the_pol, name, the_ind, value=RCSStatistics(mean=val))
        if has_noise:
            out.noise = RCSStatistics(mean=noise_val)
        return out

    def get_rcs_value(the_stats, the_pol, name, the_ind):
        def make_stat_entry(vals):
            the_count = vals['count']
            if the_count == 0:
                the_mean = float('NaN')
                the_var = float('NaN')
            else:
                the_mean = float(vals['total']/the_count)
                the_var = vals['total2']/the_count - the_mean*the_mean
                if the_var < 0:
                    the_var = 0  # to avoid floating point errors
            return RCSStatistics(
                mean=the_mean, std=float(numpy.sqrt(the_var)), min=float(vals['min']), max=float(vals['max']))

        the_entry = the_stats[name]
        noise_value = the_entry.get('noise', None)

        out = RCSValue(the_pol, name, the_ind)
        out.value = make_stat_entry(the_entry['value'])
        if noise_value is not None:
            out.noise = make_stat_entry(noise_value)
        return out

    # verify that all footprint are identical
    data_sizes = reader.get_data_size_as_tuple()
    if len(data_sizes) > 1:
        for entry in data_sizes[1:]:
            if entry != data_sizes[0]:
                raise ValueError('Each image index must have identical size')
    data_size = data_sizes[0]

    if isinstance(polygon, Polygon):
        polygons = [polygon, ]
    elif isinstance(polygon, MultiPolygon):
        polygons = polygon.polygons
    else:
        raise TypeError('polygon must be a Polygon or MultiPolygon, got type {}'.format(type(polygon)))

    sicds = reader.get_sicds_as_tuple()
    radiometric_signature = None
    for sicd in sicds:
        if radiometric_signature is None:
            radiometric_signature = evaluate_sicd(sicd)
        elif radiometric_signature != evaluate_sicd(sicd):
            raise ValueError('All sicds in the reader must have compatible Radiometric definition')
    has_radiometric, has_noise = radiometric_signature

    # construct the statistics values - first/second moments and max/min
    stat_values = [get_stat_entries() for _ in sicds]

    for polygon in polygons:
        row_bounds, col_bounds, mask = _get_polygon_bounds(polygon, data_size)
        if not numpy.any(mask):
            continue

        for i, sicd in enumerate(sicds):
            current_stat_entries = stat_values[i]

            # define the pixel power array for the given polygon and image index
            data = reader[row_bounds[0]:row_bounds[1], col_bounds[0]:col_bounds[1], i][mask]
            data = data.real * data.real + data.imag * data.imag  # get pixel power

            # define the pixel power statistics
            calculate_statistics(data, current_stat_entries['PixelPower']['value'])

            if has_radiometric:
                noise_poly = sicd.Radiometric.NoiseLevel.NoisePoly if has_noise else None
                # construct the physical coordinate arrays
                row_array = numpy.arange(row_bounds[0], row_bounds[1], 1, dtype=numpy.int32)
                x_array = get_im_physical_coords(row_array, sicd.Grid, sicd.ImageData, 'Row')
                col_array = numpy.arange(col_bounds[0], col_bounds[1], 1, dtype=numpy.int32)
                y_array = get_im_physical_coords(col_array, sicd.Grid, sicd.ImageData, 'Col')
                yarr, xarr = numpy.meshgrid(y_array, x_array)
                xarr = xarr[mask]
                yarr = yarr[mask]

                noise_power = numpy.exp(numpy.log(10)*0.1*noise_poly(xarr, yarr)) if has_noise else None
                if has_noise:
                    # add the noise statistics for the pixel power
                    calculate_statistics(noise_power, current_stat_entries['PixelPower']['noise'])

                for units_name, rcs_poly_name in DEFAULT_NAME_MAPPING.items():
                    the_poly = getattr(sicd.Radiometric, rcs_poly_name)
                    sf_data = the_poly(xarr, yarr)
                    calculate_statistics(sf_data*data, current_stat_entries[units_name]['value'])
                    if has_noise:
                        calculate_statistics(sf_data*noise_power, current_stat_entries[units_name]['noise'])

    # convert this collection of raw data to the RCSStatistics collection
    rcs_values = RCSValueCollection()
    for i, sicd in enumerate(sicds):
        polarization = sicd.get_processed_polarization()
        oversample = sicd.Grid.Row.get_oversample_rate()*sicd.Grid.Col.get_oversample_rate()
        raw_stats = stat_values[i]
        # create the total rcs/power entry
        rcs_values.insert_new_element(get_total_rcs(raw_stats, polarization, i, oversample))
        rcs_values.insert_new_element(get_rcs_value(raw_stats, polarization, 'PixelPower', i))
        if has_radiometric:
            for the_units in DEFAULT_NAME_MAPPING.keys():
                rcs_values.insert_new_element(get_rcs_value(raw_stats, polarization, the_units, i))
    return rcs_values


class RCSStatistics(Jsonable):
    __slots__ = ('mean', 'std', 'max', 'min')
    _type = 'RCSStatistics'

    def __init__(self, mean=None, std=None, max=None, min=None):
        """

        Parameters
        ----------
        mean : None|float
            All values are assumed the be stored here in units of power
        std : None|float
            All values are assumed the be stored here in units of power
        max : None|float
            All values are assumed the be stored here in units of power
        min : None|float
            All values are assumed the be stored here in units of power
        """

        if mean is not None:
            mean = float(mean)
        if std is not None:
            std = float(std)
        if max is not None:
            max = float(max)
        if min is not None:
            min = float(min)

        self.mean = mean  # type: Union[None, float]
        self.std = std  # type: Union[None, float]
        self.max = max  # type: Union[None, float]
        self.min = min  # type: Union[None, float]

    @classmethod
    def from_dict(cls, the_json):
        typ = the_json['type']
        if typ != cls._type:
            raise ValueError('RCSStatistics cannot be constructed from {}'.format(the_json))
        return cls(
            mean=the_json.get('mean', None),
            std=the_json.get('std', None),
            max=the_json.get('max', None),
            min=the_json.get('min', None))

    def to_dict(self, parent_dict=None):
        if parent_dict is None:
            parent_dict = OrderedDict()
        parent_dict['type'] = self.type
        for attr in self.__slots__:
            parent_dict[attr] = getattr(self, attr)
        return parent_dict

    def get_field_list(self):
        if self.mean is None:
            return '', '', '', '', ''
        else:
            mean_db_str = '' if self.mean <= 0 else '{0:0.5G}'.format(10*numpy.log10(self.mean))
            return (
                mean_db_str,
                '{0:0.5G}'.format(self.mean),
                '{0:0.5G}'.format(self.std) if self.std is not None else '',
                '{0:0.5G}'.format(self.min) if self.min is not None else '',
                '{0:0.5G}'.format(self.max) if self.max is not None else '',
            )


class RCSValue(Jsonable):
    """
    The collection of RCSStatistics elements.
    """

    __slots__ = ('polarization', 'units', '_index', '_value', '_noise')
    _type = 'RCSValue'

    def __init__(self, polarization, units, index, value=None, noise=None):
        """

        Parameters
        ----------
        polarization : str
        units: str
        index : int
        value : None|RCSStatistics
        noise : None|RCSStatistics
        """
        self._value = None
        self._noise = None
        self._index = None
        self.polarization = polarization
        self.units = units
        self.index = index
        self.value = value
        self.noise = noise

    @property
    def value(self):
        """
        None|RCSStatistics: The value
        """

        return self._value

    @value.setter
    def value(self, val):
        if isinstance(val, dict):
            val = RCSStatistics.from_dict(val)
        if not (val is None or isinstance(val, RCSStatistics)):
            raise TypeError('Got incompatible input for value')
        self._value = val

    @property
    def index(self):
        """
        int: The image index to which this applies
        """

        return self._index

    @index.setter
    def index(self, value):
        if value is None:
            value = 0
        self._index = int(value)

    @property
    def noise(self):
        """
        None|RCSStatistics: The noise
        """

        return self._noise

    @noise.setter
    def noise(self, val):
        if isinstance(val, dict):
            val = RCSStatistics.from_dict(val)
        if not (val is None or isinstance(val, RCSStatistics)):
            raise TypeError('Got incompatible input for noise')
        self._noise = val

    @classmethod
    def from_dict(cls, the_json):  # type: (dict) -> RCSValue
        typ = the_json['type']
        if typ != cls._type:
            raise ValueError('RCSValue cannot be constructed from {}'.format(the_json))
        return cls(
            the_json.get('polarization', None),
            the_json.get('units', None),
            the_json.get('index', None),
            value=the_json.get('value', None),
            noise=the_json.get('noise', None))

    def to_dict(self, parent_dict=None):
        if parent_dict is None:
            parent_dict = OrderedDict()
        parent_dict['type'] = self.type
        parent_dict['polarization'] = self.polarization
        parent_dict['units'] = self.units
        parent_dict['index'] = self.index
        if self.value is not None:
            parent_dict['value'] = self.value.to_dict()
        if self.noise is not None:
            parent_dict['noise'] = self.noise.to_dict()
        return parent_dict


class RCSValueCollection(Jsonable):
    """
    A specific type for the AnnotationProperties.parameters
    """

    __slots__ = ('_pixel_count', '_elements')
    _type = 'RCSValueCollection'

    def __init__(self, pixel_count=None, elements=None):
        """

        Parameters
        ----------
        pixel_count : None|int
        elements : None|List[RCSValue|dict]
        """

        self._pixel_count = None
        self._elements = []

        self.pixel_count = pixel_count
        self.elements = elements

    def __len__(self):
        return len(self._elements)

    def __getitem__(self, item):
        # type: (Union[int, str]) -> Union[None, RCSValue]
        return self._elements[item]

    @property
    def pixel_count(self):
        # type: () -> Union[None, int]
        """
        None|int: The number of integer pixel grid elements contained in the interior
        of the associated geometry element.
        """

        return self._pixel_count

    @pixel_count.setter
    def pixel_count(self, value):
        if value is None:
            self._pixel_count = None
            return
        if not isinstance(value, int):
            value = int(value)
        self._pixel_count = value

    @property
    def elements(self):
        # type: () -> Union[None, List[RCSValue]]
        """
        List[RCSValue]: The RCSValue elements.
        """

        return self._elements

    @elements.setter
    def elements(self, elements):
        if elements is None:
            self._elements = []
            return

        if not isinstance(elements, list):
            raise TypeError('elements must be a list of RCSValue elements')
        self._elements = []
        for element in elements:
            self.insert_new_element(element)

    def insert_new_element(self, element):
        """
        Inserts an element at the end of the elements list.

        Parameters
        ----------
        element : RCSValue
        """

        if isinstance(element, dict):
            element = RCSValue.from_dict(element)
        if not isinstance(element, RCSValue):
            raise TypeError('element must be an RCSValue instance')
        self._elements.append(element)

    @classmethod
    def from_dict(cls, the_json):
        # type: (dict) -> RCSValueCollection

        typ = the_json['type']
        if typ != cls._type:
            raise ValueError('RCSValueCollection cannot be constructed from {}'.format(the_json))
        return cls(
            pixel_count=the_json.get('pixel_count', None), elements=the_json.get('elements', None))

    def to_dict(self, parent_dict=None):
        if parent_dict is None:
            parent_dict = OrderedDict()
        parent_dict['type'] = self.type
        parent_dict['pixel_count'] = self.pixel_count
        if len(self._elements) > 0:
            parent_dict['elements'] = [entry.to_dict() for entry in self._elements]
        return parent_dict


class RCSProperties(AnnotationProperties):
    _type = 'RCSProperties'

    @property
    def parameters(self):
        """
        RCSValueCollection: The parameters
        """

        return self._parameters

    @parameters.setter
    def parameters(self, value):
        if value is None:
            self._parameters = RCSValueCollection()
            return

        if isinstance(value, dict):
            value = RCSValueCollection.from_dict(value)
        if not isinstance(value, RCSValueCollection):
            raise TypeError('Got unexpected type for parameters')
        self._parameters = value


class RCSFeature(AnnotationFeature):
    """
    A specific extension of the Feature class which has the properties attribute
    populated with RCSValueCollection instance.
    """
    _allowed_geometries = (Polygon, MultiPolygon)

    @property
    def properties(self):
        # type: () -> RCSProperties
        """
        The properties.

        Returns
        -------
        RCSProperties
        """

        return self._properties

    @properties.setter
    def properties(self, properties):
        if properties is None:
            self._properties = RCSProperties()
        elif isinstance(properties, RCSProperties):
            self._properties = properties
        elif isinstance(properties, dict):
            self._properties = RCSProperties.from_dict(properties)
        else:
            raise TypeError('properties must be an RCSProperties')

    def set_rcs_parameters_from_reader(self, reader):
        """
        Given a SICD type reader construct the `RCSValueCollection` and set that
        as the properties.parameters value.

        Parameters
        ----------
        reader : SICDTypeReader
        """

        if self.geometry is None or self.geometry_count == 0:
            self.properties.parameters = None
        else:
            # noinspection PyTypeChecker
            self.properties.parameters = create_rcs_value_collection_for_reader(
                reader, self.geometry)


class RCSCollection(AnnotationCollection):
    """
    A specific extension of the AnnotationCollection class which has that the
    features are RCSFeature instances.
    """

    @property
    def features(self):
        """
        The features list.

        Returns
        -------
        List[RCSFeature]
        """

        return self._features

    @features.setter
    def features(self, features):
        if features is None:
            self._features = None
            self._feature_dict = None
            return

        if not isinstance(features, list):
            raise TypeError('features must be a list of RCSFeatures. Got {}'.format(type(features)))

        for entry in features:
            self.add_feature(entry)

    def add_feature(self, feature):
        """
        Add an annotation.

        Parameters
        ----------
        feature : RCSFeature|dict
        """

        if isinstance(feature, dict):
            feature = RCSFeature.from_dict(feature)

        if not isinstance(feature, RCSFeature):
            raise TypeError('This requires an RCSFeature instance, got {}'.format(type(feature)))

        if self._features is None:
            self._feature_dict = {feature.uid: 0}
            self._features = [feature, ]
        else:
            self._feature_dict[feature.uid] = len(self._features)
            self._features.append(feature)

    def __getitem__(self, item):
        # type: (Any) -> Union[RCSFeature, List[RCSFeature]]
        if self._features is None:
            raise StopIteration

        if isinstance(item, str):
            index = self._feature_dict[item]
            return self._features[index]
        return self._features[item]


###########
# serialized file object

class FileRCSCollection(FileAnnotationCollection):
    """
    An collection of RCS statistics elements.
    """
    _type = 'FileRCSCollection'

    def __init__(self, version=None, annotations=None, image_file_name=None,
                 image_id=None, core_name=None):
        if version is None:
            version = _RCS_VERSION

        FileAnnotationCollection.__init__(
            self, version=version, annotations=annotations, image_file_name=image_file_name,
            image_id=image_id, core_name=core_name)

    @property
    def annotations(self):
        """
        The annotations.

        Returns
        -------
        RCSCollection
        """

        return self._annotations

    @annotations.setter
    def annotations(self, annotations):
        # type: (Union[None, RCSCollection, dict]) -> None
        if annotations is None:
            self._annotations = None
            return

        if isinstance(annotations, RCSCollection):
            self._annotations = annotations
        elif isinstance(annotations, dict):
            self._annotations = RCSCollection.from_dict(annotations)
        else:
            raise TypeError(
                'annotations must be an RCSCollection. Got type {}'.format(type(annotations)))

    def add_annotation(self, annotation):
        """
        Add an annotation.

        Parameters
        ----------
        annotation : RCSFeature
            The prospective annotation.
        """

        if isinstance(annotation, dict):
            annotation = RCSFeature.from_dict(annotation)
        if not isinstance(annotation, RCSFeature):
            raise TypeError('This requires an RCSFeature instance. Got {}'.format(type(annotation)))

        if self._annotations is None:
            self._annotations = RCSCollection()

        self._annotations.add_feature(annotation)

    def delete_annotation(self, annotation_id):
        """
        Deletes the annotation associated with the given id.

        Parameters
        ----------
        annotation_id : str
        """

        del self._annotations[annotation_id]

    @classmethod
    def from_file(cls, file_name):
        """
        Read from (json) file.

        Parameters
        ----------
        file_name : str

        Returns
        -------
        FileRCSCollection
        """

        with open(file_name, 'r') as fi:
            the_dict = json.load(fi)
        return cls.from_dict(the_dict)

    @classmethod
    def from_dict(cls, the_dict):
        """
        Define from a dictionary representation.

        Parameters
        ----------
        the_dict : dict

        Returns
        -------
        FileRCSCollection
        """

        if not isinstance(the_dict, dict):
            raise TypeError('This requires a dict. Got type {}'.format(type(the_dict)))

        typ = the_dict.get('type', 'NONE')
        if typ != cls._type:
            raise ValueError('FileRCSCollection cannot be constructed from the input dictionary')

        return cls(
            version=the_dict.get('version', 'UNKNOWN'),
            annotations=the_dict.get('annotations', None),
            image_file_name=the_dict.get('image_file_name', None),
            image_id=the_dict.get('image_id', None),
            core_name=the_dict.get('core_name', None))
