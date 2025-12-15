from collections import OrderedDict
import pytest
import re
import unittest
from uuid import UUID, uuid4

from sarpy.annotation.base import GeometryProperties, AnnotationProperties, AnnotationFeature, AnnotationCollection, FileAnnotationCollection
from sarpy.geometry.geometry_elements import GeometryCollection, GeometryObject, Jsonable, Point, Polygon

class test_geometryproperties(unittest.TestCase):
    def test_geometryproperties_default_uid(self):
        # confirms that a uuid is generated if one isn't provided
        geom = GeometryProperties()
        self.assertIsInstance(geom.uid, str)

        try:
            # checks if uuid generated is a valid uuid
            UUID(geom.uid)
        except:
            pytest.fail("The generated uid is not a valid UUID string")

    def test_geometryproperties_custom_uid(self):
        # confirms that that the uid setter works as expected
        custom_uid = "abcd"
        geom = GeometryProperties(uid=custom_uid)

        self.assertEqual(geom.uid, custom_uid)

    def test_geometryproperties_invalid_uid(self):
        custom_uid = 1234

        with pytest.raises(TypeError, 
                            match = re.escape("uid must be a string, got <class 'int'>")):
            geom = GeometryProperties(uid=custom_uid)

    def test_geometryproperties_default_name(self):
        geom = GeometryProperties()

        self.assertIsNone(geom.name)

    def test_geometryproperties_custom_name(self):
        custom_name = "abcd"
        geom = GeometryProperties(name=custom_name)

        self.assertEqual(geom.name, custom_name)

    def test_geometryproperties_invalid_name(self):
        custom_name = 1234

        with pytest.raises(TypeError, 
                            match = re.escape("Got unexpected type of <class 'int'> for name")):
            geom = GeometryProperties(name=custom_name)

    def test_geometryproperties_default_color(self):
        geom = GeometryProperties()

        self.assertIsNone(geom.color)

    def test_geometryproperties_custom_color(self):
        custom_color = "abcd"
        geom = GeometryProperties(color=custom_color)

        self.assertEqual(geom.color, custom_color)

    def test_geometryproperties_invalid_color(self):
        custom_color = 1234

        with pytest.raises(TypeError, 
                            match = re.escape("Got unexpected type of <class 'int'> for color")):
            geom = GeometryProperties(color=custom_color)

    def test_geometryproperties_all_fields(self):
        # makes sure that GeomeryProperties is able to be initialized properly with all three parameters passed
        geom = GeometryProperties(uid="abcd", name="efgh", color="blue")

        self.assertEqual(geom.uid, "abcd")
        self.assertEqual(geom.name, "efgh")
        self.assertEqual(geom.color, "blue")

    def test_geometryproperties_from_dict_valid_data(self):
        data = {
            "type": "GeometryProperties",
            "uid": "abcd",
            "name": "efgh",
            "color": "red"
        }

        geom = GeometryProperties.from_dict(data)

        self.assertIsInstance(geom, GeometryProperties)
        self.assertEqual(geom.uid, "abcd")
        self.assertEqual(geom.name, "efgh")
        self.assertEqual(geom.color, "red")

    def test_geometry_properties_from_dict_int(self):
        with pytest.raises(TypeError,
                            match = re.escape("This requires a dict. Got type <class 'int'>")):
            geom = GeometryProperties.from_dict(1234)
        
    def test_geometryproperties_from_dict_missing_type(self):
        # confirms that an error is raised if the dict is missing type
        bad_data = {
            "uid": "abcd",
            "name": "efgh",
            "color": "red"
        }
        
        with pytest.raises(KeyError,
                            match = re.escape("the json requires the field 'type'")):
            geom = GeometryProperties.from_dict(bad_data)
    
    def test_geometryproperties_from_dict_wrong_type_value(self):
        # confirms that an error is raised if the dict is missing type
        bad_data = {
            "type": "incorrect_string",
            "uid": "abcd",
            "name": "efgh",
            "color": "red"
        }

        with pytest.raises(ValueError, 
                            match = re.escape("GeometryProperties cannot be constructed from incorrect_string, expecting GeometryProperties")):
            geom = GeometryProperties.from_dict(bad_data)

    def test_geometryproperties_to_dict_geometryproperties_w_values(self):
        # confirms that the to_dict() function works as expected and includes all of the attributes in a dictionary 
        geom_dict = GeometryProperties(uid="abcd", name="efgh", color="blue").to_dict()

        self.assertIsInstance(geom_dict, dict)
        self.assertEqual(geom_dict["uid"], "abcd")
        self.assertEqual(geom_dict["name"], "efgh")
        self.assertEqual(geom_dict["color"], "blue")

    def test_geometryproperties_to_dict_exclude_none_fields(self):
        # confirms that None fields are excluded
        geom_dict = GeometryProperties(uid="abcd", name=None, color=None).to_dict()
        
        self.assertIsInstance(geom_dict, dict)
        self.assertEqual(geom_dict["uid"], "abcd")
        self.assertNotIn("name", geom_dict)
        self.assertNotIn("color", geom_dict)

    def test_geometryproperties_to_dict_w_parent_dict(self):
        # confirms that the to_dict() function works with a parent_dict passed
        parent_dict = OrderedDict([("key", "value")])

        geom_dict = GeometryProperties(uid="abcd", name="efgh", color="blue").to_dict(parent_dict)

        self.assertEqual(geom_dict["key"], "value")
        self.assertEqual(geom_dict["uid"], "abcd")
        self.assertEqual(geom_dict["name"], "efgh")
        self.assertEqual(geom_dict["color"], "blue")

class test_annotationproperties(unittest.TestCase):
    def setUp(self):
        self.geometryproperties_obj = GeometryProperties(uid="abcd", name="efgh", color="blue")

        self.test_parameter_point = Point((0,0))

        self.annotation_properties_obj = AnnotationProperties(
            name = "annotation1",
            description = "abcd",
            directory = "path/folder",
            geometry_properties = [self.geometryproperties_obj],
            parameters = self.test_parameter_point
        )

        self.geometry_data = {"type": "GeometryProperties",
                    "uid": str(uuid4()),
                    "name": "apple",
                    "color": "red"}

    def test_annotationproperties_default_initialization(self):
        obj = AnnotationProperties()

        self.assertIsNone(obj.name)
        self.assertIsNone(obj.description)
        self.assertIsNone(obj.directory)
        self.assertEqual(obj.geometry_properties, [])
        self.assertIsNone(obj.parameters)

    def test_annotationproperties_initialization_all_fields(self):
        obj = AnnotationProperties(
            name = "annotation1",
            description = "abcd",
            directory = "path/folder",
            geometry_properties = [self.geometryproperties_obj],
            parameters = self.test_parameter_point
        )

        self.assertEqual(obj.name, "annotation1")
        self.assertEqual(obj.description, "abcd")
        self.assertEqual(obj.directory, "path/folder")
        self.assertIsInstance(obj.geometry_properties, list)

        for i, each_geometry_property in enumerate(obj.geometry_properties, 0):
            self.assertIsInstance(each_geometry_property, GeometryProperties,
                                  msg = f'Item at index {i} is of type {type(each_geometry_property)}, expected GeometryProperties')
        
        self.assertIsInstance(obj.parameters, Jsonable)
    
    def test_annotation_properties_invalid_name(self):
        with pytest.raises(TypeError, 
                            match = re.escape("Got unexpected value of type <class 'int'> for name")):
            obj = AnnotationProperties(
            name = 1234,
            description = "abcd",
            directory = "path/folder",
            geometry_properties = [self.geometryproperties_obj],
            parameters = self.test_parameter_point
        )

    def test_annotationproperties_invalid_description(self):
        with pytest.raises(TypeError, 
                            match = re.escape("Got unexpected value of type <class 'int'> for description")):
            obj = AnnotationProperties(
                name = "annotation1",
                description = 1234,
                directory = "path/folder",
                geometry_properties = [self.geometryproperties_obj],
                parameters = self.test_parameter_point
            )
        
    def test_annotationproperties_invalid_directory(self):
        with pytest.raises(TypeError, 
                            match = re.escape("Got unexpected value of type <class 'int'> for directory")):
            obj = AnnotationProperties(
                name = "annotation1",
                description = "abcd",
                directory = 1234,
                geometry_properties = [self.geometryproperties_obj],
                parameters = self.test_parameter_point
            )
    
    def test_annotationproperties_invalid_geometry(self):
        with pytest.raises(TypeError, 
                            match = re.escape("Got unexpected value of type <class 'str'> for geometry")):
            obj = AnnotationProperties(
                name = "annotation1",
                description = "abcd",
                directory = "path/folder",
                geometry_properties = ['abcd'],
                parameters = self.test_parameter_point
            )

    def test_annotationproperties_invalid_properties(self):
        with pytest.raises(TypeError, 
                            match = re.escape("Got unexpected value of type <class 'str'> for parameters")):
            obj = AnnotationProperties(
                name = "annotation1",
                description = "abcd",
                directory = "path/folder",
                geometry_properties = [self.geometryproperties_obj],
                parameters = "abcd"
            )

    def test_annotationproperties_geometryproperties_setter(self):
        obj = AnnotationProperties()
        obj.geometry_properties = [self.geometryproperties_obj, self.geometryproperties_obj]

        self.assertEqual(len(obj.geometry_properties), 2)

    def test_annotationproperties_geometryproperties_setter_type_error(self):
        obj = AnnotationProperties()

        with pytest.raises(TypeError, 
                match = re.escape("Got unexpected value of type <class 'str'> for geometry properties")):
            obj.geometry_properties = "abcd"

    def test_annotationproperties_add_geometry_property(self):
        obj = self.annotation_properties_obj
        geom_prop = GeometryProperties(uid="efgh", name="ijkl", color="green")

        # check length of geometry_properties before and after the new property was added
        self.assertEqual(len(obj.geometry_properties), 1)

        obj.add_geometry_property(geom_prop)
        
        # confirms that the property was successfully added since the length increased
        self.assertEqual(len(obj.geometry_properties), 2)
        self.assertEqual(obj.geometry_properties[1], geom_prop)
    
    def test_annotationproperties_add_geometry_property_invalid_type(self):
        with pytest.raises(TypeError, 
                            match = re.escape("Got unexpected value of type <class 'str'> for geometry properties")):
            obj = self.annotation_properties_obj
            obj.add_geometry_property("abcd")

    def test_annotationproperties_get_geometry_property(self):
        obj = self.annotation_properties_obj
        geom_properties = obj.get_geometry_property("abcd")

        self.assertEqual(geom_properties.color, "blue")

    def test_annotationproperties_get_property_and_index_by_index(self):
        property, index = self.annotation_properties_obj.get_geometry_property_and_index(0)

        self.assertEqual(property, self.geometryproperties_obj)
        self.assertEqual(index, 0)

    def test_annotationproperties_get_property_and_index_by_uid(self):
        property, index = self.annotation_properties_obj.get_geometry_property_and_index("abcd")

        self.assertEqual(property, self.geometryproperties_obj)
        self.assertEqual(index, 0)
        
    def test_annotationproperties_get_property_and_index_invalid_key(self):
        with pytest.raises(KeyError, 
                            match = re.escape("Got unrecognized geometry key `3.14`")):
           property, index = self.annotation_properties_obj.get_geometry_property_and_index(3.14)

    def test_annotationproperties_from_dict_valid_input(self):
        data = {
            "type": "AnnotationProperties",
            "name": "annotation1",
            "description": "abcd",
            "directory": "path/folder",
            "geometry_properties": [self.geometry_data],
            "parameters": None
        }
        
        obj = AnnotationProperties.from_dict(data)

        self.assertIsInstance(obj, AnnotationProperties)
        self.assertEqual(obj.name, "annotation1")
        self.assertEqual(obj.description, "abcd")
        self.assertEqual(obj.directory, "path/folder")
        self.assertIsInstance(obj.geometry_properties, list)
        for i, each_geometry_property in enumerate(obj.geometry_properties, 0):
            self.assertIsInstance(each_geometry_property, GeometryProperties,
                                  msg = f'Item at index {i} is of type {type(each_geometry_property)}, expected GeometryProperties')
    
    def test_annotationproperties_from_dict_invalid_type_value(self):
        with pytest.raises(ValueError, 
                            match = re.escape('AnnotationProperties cannot be constructed from abcd, expecting AnnotationProperties')):
            # type "abcd" is invalid
            data = {
                "type": "abcd",
                "name": "annotation1",
                "description": "abcd",
                "directory": "path/folder",
                "geometry_properties": [self.geometry_data],
                "parameters": None
            }
            
            obj = AnnotationProperties.from_dict(data)

    def test_annotationproperties_from_dict_missing_type_value(self):
        with pytest.raises(KeyError, 
                            match = re.escape("the json requires the field 'type'")):
            data = {
                "name": "annotation1",
                "description": "abcd",
                "directory": "path/folder",
                "geometry_properties": [self.geometry_data],
                "parameters": None
            }
            
            obj = AnnotationProperties.from_dict(data)

    def test_annotationproperties_from_dict_int(self):
        with pytest.raises(TypeError, 
                            match = re.escape("This requires a dict. Got type <class 'int'>")):
            obj = AnnotationProperties.from_dict(1234)

    def test_annotationproperties_to_dict_valid_input(self):
        obj = self.annotation_properties_obj.to_dict()

        self.assertEqual(obj["type"], "AnnotationProperties")
        self.assertEqual(obj["name"], "annotation1")
        self.assertEqual(obj["description"], "abcd")
        self.assertEqual(obj["directory"], "path/folder")
        self.assertEqual(obj["parameters"]["type"], "Point")
    
    def test_annotationproperties_to_dict_w_parent_dict(self):
        parent_dict = {"key": "value"}

        obj = self.annotation_properties_obj.to_dict(parent_dict)
        
        # this function should be adding annotationproperties to the existing parent_dict object
        self.assertEqual(obj, parent_dict)
    
        # confirms the rest of the values from self.annotation_properties_obj were added to obj
        self.assertEqual(obj["key"], "value")
        self.assertEqual(obj["name"], "annotation1")
        self.assertEqual(obj["description"], "abcd")
        self.assertEqual(obj["directory"], "path/folder")
        self.assertEqual(obj["parameters"]["type"], "Point")
    
    def test_annotationproperties_replicate_valid_input(self):
        obj = self.annotation_properties_obj
        replica = obj.replicate()

        self.assertIsInstance(replica, AnnotationProperties)
        self.assertEqual(replica.name, obj.name)
        self.assertEqual(replica.description, obj.description)
        self.assertEqual(replica.directory, obj.directory)

class test_annotationfeature(unittest.TestCase):
    def setUp(self):
        self.geometryproperties_obj = GeometryProperties(uid="abcd", name="efgh", color="blue")

        self.test_parameter_point = Point((0,0))

        self.annotation_properties_obj = AnnotationProperties(
            name = "annotation1",
            description = "abcd",
            directory = "path/folder",
            geometry_properties = [self.geometryproperties_obj],
            parameters = self.test_parameter_point
        )

        self.geom_dict = {
                        "type": "Point",
                        "coordinates": [0, 0]
                     }
        self.geometry_obj = GeometryObject.from_dict(self.geom_dict)
    
        self.annotation_feature_obj = AnnotationFeature(geometry=self.geometry_obj, properties=self.annotation_properties_obj)

    def test_annotationfeature_initialization(self):
        obj = AnnotationFeature(geometry=self.geometry_obj, properties=self.annotation_properties_obj)

        self.assertEqual(obj.geometry, self.geometry_obj)
    
        self.assertEqual(obj.properties.name, "annotation1")
        self.assertEqual(obj.properties.description, "abcd")
        self.assertEqual(obj.properties.directory, "path/folder")
    
    def test_annotationfeature_set_properties_none(self):
        obj = self.annotation_feature_obj
        obj.properties = None

        self.assertEqual(obj.properties.name, None)
        self.assertEqual(obj.properties.description, None)
        self.assertEqual(obj.properties.directory, None)

    def test_annotationfeature_set_properties_obj(self):
        obj = self.annotation_feature_obj

        new_properties = AnnotationProperties(name="annotation2", description="new description", directory="new path")
        obj.properties = new_properties

        self.assertEqual(obj.properties.name, "annotation2")
        self.assertEqual(obj.properties.description, "new description")
        self.assertEqual(obj.properties.directory, "new path")

    def test_annotationfeature_set_properties_from_dict(self):
        obj = self.annotation_feature_obj

        properties_dict = {
            "type": "AnnotationProperties",
            "name": "annotation3",
            "description": "new description",
            "directory": "new path"
        }
        
        obj.properties = properties_dict

        self.assertEqual(obj.properties.name, "annotation3")
        self.assertEqual(obj.properties.description, "new description")
        self.assertEqual(obj.properties.directory, "new path")

    def test_annotationfeature_set_properties_invalid_type(self):
        obj = self.annotation_feature_obj
        with pytest.raises(TypeError, 
                            match = re.escape("Got an unexpected type for properties attribute of type <class 'str'>")):
            obj.properties = "abcd"
    
    def test_annotationfeature_get_name_none(self):
        obj = AnnotationFeature()

        # confirms that a uid is generated even if there's no properties or properties.name
        self.assertIsNotNone(obj.get_name())
        self.assertTrue(UUID(obj.uid, version=4))

    def test_annotationfeature_get_name(self):
        obj = self.annotation_feature_obj

        self.assertEqual(obj.get_name(), "annotation1")
    
    def test_annotationfeature_set_geometry_geometryobject(self):
        obj = self.annotation_feature_obj

        geom_dict = {
                        "type": "Point",
                        "coordinates": [2048, 2048]
                     }
        
        geom_object = GeometryObject.from_dict(geom_dict)
        obj.geometry = geom_object

        self.assertEqual(obj.geometry, geom_object)

    def test_annotationfeature_set_geometry_dict(self):
        obj = self.annotation_feature_obj

        geom_dict = {
                        "type": "Point",
                        "coordinates": [4096, 4096]
                     }
        
        obj.geometry = geom_dict

        self.assertListEqual(obj.geometry.coordinates.tolist(), GeometryObject.from_dict(geom_dict).coordinates.tolist())

    def test_annotationfeature_invalid_set_geometry(self):
        with pytest.raises(TypeError, 
                            match = re.escape("geometry must be an instance of Geometry, got `<class 'str'>`")):
            obj = self.annotation_feature_obj
            geom_dict = "abcd"
            obj.geometry = geom_dict
    
    def test_annotationfeature_geometry_count_none(self):
        obj = AnnotationFeature()

        self.assertEqual(obj.geometry_count, 0)

    def test_annotationfeature_geometry_count_collection(self):
        p1 = Point([1, 1])
        p2 = Point([2, 2])
        p3 = Point([3, 3])

        geom_collection = GeometryCollection([p1, p2, p3])

        obj = self.annotation_feature_obj
        obj.geometry = geom_collection

        self.assertEqual(obj.geometry_count, 3)        

    def test_annotationfeature_geometry_count_non_collection(self):
        obj = self.annotation_feature_obj

        self.assertEqual(obj.geometry_count, 1)

    def test_annotationfeature_get_geometry_name(self):
        obj = self.annotation_feature_obj

        self.assertEqual(obj.get_geometry_name(0), "efgh")
    
    def test_annotationfeature_get_geometry_property(self):
        obj = self.annotation_feature_obj
        
        self.assertEqual(obj.get_geometry_property("abcd"), self.geometryproperties_obj)
    
    def test_annotationfeature_get_geometry_point_and_index(self):
        obj = self.annotation_feature_obj

        self.assertEqual(obj.get_geometry_property_and_index(0), (self.geometryproperties_obj, 0))
    
    def test_annotationfeature_get_geometry_and_geometry_properties(self):
        obj = self.annotation_feature_obj

        self.assertEqual(obj.get_geometry_and_geometry_properties(0), (self.geometry_obj, self.geometryproperties_obj))
    
    def test_annotationfeature_get_geometry_and_geometry_properties_collection(self):
        p1 = Point([1, 1])
        p2 = Point([2, 2])
        p3 = Point([3, 3])

        geom_collection = GeometryCollection([p1, p2, p3])

        obj = self.annotation_feature_obj
        obj.geometry = geom_collection

        self.assertEqual(obj.get_geometry_and_geometry_properties(0), (p1, self.geometryproperties_obj))

    def test_annotationfeature_get_geometry_and_geometry_properties_invalid_geometry(self):
        obj = AnnotationFeature()
        
        with pytest.raises(ValueError, match = re.escape("No geometry defined")):
            obj.get_geometry_and_geometry_properties(0)

    def test_annotationfeature_get_geometry_and_geometry_properties_invalid_geometry_index(self):
        obj = self.annotation_feature_obj

        with pytest.raises(KeyError, match = re.escape("invalid geometry index")):
            obj.get_geometry_and_geometry_properties(-1)

    def test_annotationfeature_get_geometry_element(self):
        obj = self.annotation_feature_obj

        self.assertEqual(obj.get_geometry_element(0), self.geometry_obj)
    
    def test_annotationfeature_validate_geometry_element_none(self):
        obj = AnnotationFeature()

        self.assertIsNone(obj._validate_geometry_element(None))

    def test_annotationfeature_validate_geometry_element(self):
        obj = self.annotation_feature_obj
        self.assertEqual(obj._validate_geometry_element(self.geometry_obj), self.geometry_obj)
    
    def test_annotationfeature_validate_geometry_element_non_geometry_instance(self):
        obj = self.annotation_feature_obj
        
        with pytest.raises(TypeError, match = re.escape("geometry must be an instance of Geometry base class. Got <class 'str'>")):
            obj._validate_geometry_element("abc")

    def test_annotationfeature_validate_geometry_element_type_error(self):
        obj = self.annotation_feature_obj

        # setting allowed geometries to only Polygon to force the TypeError
        obj._allowed_geometries = {Polygon}

        with pytest.raises(TypeError, match = re.escape('geometry (Point(**{\n "type": "Point",\n "coordinates": [\n  0.0,\n  0.0\n ]\n})) is not of one of the allowed types ({<class \'sarpy.geometry.geometry_elements.Polygon\'>})')):
            obj._validate_geometry_element(self.geometry_obj)
    
    def test_annotationfeature_add_geometry_element_none(self):
        obj = AnnotationFeature()

        # confirms the geometry_count is initially 0
        self.assertEqual(obj.geometry_count, 0)

        obj.add_geometry_element(self.geometry_obj)

        # confirms that a None AnnotationProperty instance is generated by checking the attributes
        self.assertIsNone(obj.properties.name)
        self.assertIsNone(obj.properties.description)
        self.assertIsNone(obj.properties.directory)

        # confirms that the geometry element got added to the newly generated AnnotationProperty of this AnnotationFeature instance
        self.assertEqual(obj.geometry_count, 1)
        self.assertEqual(obj.get_geometry_element(0), self.geometry_obj)   

    def test_annotationfeature_add_geometry_element_none_properties(self):
        obj = AnnotationFeature()
        obj.properties = None

        obj.add_geometry_element(self.geometry_obj)

        # confirms that a None AnnotationProperty instance is generated by checking the attributes
        self.assertIsNone(obj.properties.name)
        self.assertIsNone(obj.properties.description)
        self.assertIsNone(obj.properties.directory)

        self.assertEqual(obj.get_geometry_element(0), self.geometry_obj)   

    def test_annotationfeature_add_geometry_element(self):
        obj = self.annotation_feature_obj
        self.assertEqual(obj.geometry_count, 1)

        obj.add_geometry_element(self.geometry_obj)

        self.assertEqual(obj.get_geometry_element(1), self.geometry_obj)   
        self.assertEqual(obj.geometry_count, 2)

    def test_annotationfeature_add_geometry_element_invalid_geomobject(self):
        obj = self.annotation_feature_obj

        with pytest.raises(TypeError, match = re.escape("geometry must be a GeometryObject instance. Got <class 'str'>")):
            obj.add_geometry_element("abcd")
    
    def test_annotationfeature_add_geometry_element_invalid_geomproperties(self):
        obj = self.annotation_feature_obj

        with pytest.raises(TypeError, match = re.escape("properties must be a GeometryProperties instance. Got <class 'str'>")):
            obj.add_geometry_element(self.geometry_obj, "abcd")

    def test_annotationfeature_add_geometry_element_logger_warning(self):
        obj = AnnotationFeature()
        obj.geometry = GeometryCollection()
        obj.properties = AnnotationProperties()

        # add one geometry with one property: geom = 1, props = 1
        obj.add_geometry_element(self.geometry_obj, self.geometryproperties_obj)

        # remove the property to create mismatch
        obj.properties.geometry_properties = []  # props = 0

        expected_warning = "This is likely to cause problems."

        with self.assertLogs('sarpy.annotation.base', level="WARNING") as context_manager:
            # add geometry -> geom = 2, props = 0 -> mismatch -> warning
            obj.add_geometry_element(GeometryObject.from_dict(self.geom_dict))

        self.assertTrue(
            any(expected_warning in msg for msg in context_manager.output),
            "Expected geometry/property mismatch warning not logged"
        )

    def test_annotationfeature_remove_geometry_element(self):
        obj = self.annotation_feature_obj
        geom_dict = {
                        "type": "Point",
                        "coordinates": [1, 1]
                     }
        geometry_obj = GeometryObject.from_dict(geom_dict)

        # add Point(1, 1) in addition to Point(0, 0) which was already there
        obj.add_geometry_element(geometry_obj)

        self.assertEqual(obj.geometry_count, 2)

        # remove Point(0,0) which was originally in self.annotation_feature_obj before we removed it
        obj.remove_geometry_element(0)

        self.assertEqual(obj.geometry_count, 1)
        
        # confirm that Point(1, 1) is all that's left 
        self.assertEqual(obj.geometry, geometry_obj)

    def test_annotationfeature_remove_geometry_element_one_element(self):
        obj = self.annotation_feature_obj

        obj.remove_geometry_element(0)

        self.assertIsNone(obj.geometry)

        # confirms None AnnotationProperty instance by checking the attributes
        self.assertIsNone(obj.properties.name)
        self.assertIsNone(obj.properties.description)
        self.assertIsNone(obj.properties.directory)

    def test_annotationfeature_remove_geometry_element_more_than_two_elements(self):
        obj = self.annotation_feature_obj
        
        geom_dict = {
                        "type": "Point",
                        "coordinates": [1, 1]
                     }
        geometry_obj = GeometryObject.from_dict(geom_dict)

        # add two more points to make 3 total elements
        obj.add_geometry_element(geometry_obj)
        obj.add_geometry_element(geometry_obj)

        # confirms that there are 3 elements
        self.assertEqual(obj.geometry_count, 3)

        obj.remove_geometry_element(1)

        self.assertEqual(obj.geometry_count, 2)

        self.assertEqual(obj.get_geometry_element(0), self.geometry_obj)
        self.assertEqual(obj.get_geometry_element(1), geometry_obj)

    def test_annotationfeature_from_dict(self):
        features_dict = {
            "type": "AnnotationFeature",
            "geometry": {
                        "type": "Point",
                        "coordinates": [1, 1]
                     },
            "properties": {
                "type": "AnnotationProperties",
                "name": "annotation3",
                "description": "new description",
                "directory": "new path"
            }
        }

        obj = AnnotationFeature.from_dict(features_dict)

        self.assertEqual(obj.type, "AnnotationFeature")
        self.assertIsInstance(obj, AnnotationFeature)

    def test_annotationfeature_from_dict_key_error(self):
        features_dict = {
            "geometry": {
                        "type": "Point",
                        "coordinates": [1, 1]
                     },
            "properties": {
                "type": "AnnotationProperties",
                "name": "annotation3",
                "description": "new description",
                "directory": "new path"
            }
        }
        
        with pytest.raises(KeyError,
                            match = re.escape("the json requires the field 'type'")):
            AnnotationFeature.from_dict(features_dict)
    
    def test_annotationfeature_from_dict_type_error(self):
        features_dict = 1234
        
        with pytest.raises(TypeError,
                            match = re.escape("This requires a dict. Got type <class 'int'>")):
            AnnotationFeature.from_dict(features_dict)

    def test_annotationfeature_from_dict_value_error(self):
        features_dict = {
            "type": "incorrect_type",
            "geometry": {
                        "type": "Point",
                        "coordinates": [1, 1]
                     },
            "properties": {
                "type": "AnnotationProperties",
                "name": "annotation3",
                "description": "new description",
                "directory": "new path"
            }
        }
        
        with pytest.raises(ValueError,
                            match = re.escape("AnnotationFeature cannot be constructed from incorrect_type, expecting AnnotationFeature")):
            AnnotationFeature.from_dict(features_dict)

class test_annotationcollection(unittest.TestCase):
    def setUp(self):
        self.geometryproperties_obj = GeometryProperties(uid="abcd", name="efgh", color="blue")

        self.test_parameter_point = Point((0,0))

        self.annotation_properties_obj = AnnotationProperties(
            name = "annotation1",
            description = "abcd",
            directory = "path/folder",
            geometry_properties = [self.geometryproperties_obj],
            parameters = self.test_parameter_point
        )

        self.geom_dict = {
                        "type": "Point",
                        "coordinates": [0, 0]
                     }
        self.geometry_obj = GeometryObject.from_dict(self.geom_dict)
    
        self.annotation_feature_obj = AnnotationFeature(geometry=self.geometry_obj, properties=self.annotation_properties_obj)

        self.annotation_collection_obj =  AnnotationCollection(features=[self.annotation_feature_obj])

        new_annotation_properties_obj = AnnotationProperties(
            name = "annotation2",
            description = "efgh",
            directory = "path/folder2",
        )

        new_geom_dict = {
                "type": "Point",
                "coordinates": [1, 1]
                }
        new_geometry_obj = GeometryObject.from_dict(new_geom_dict)
    
        self.annotation_feature_obj2 = AnnotationFeature(geometry=new_geometry_obj, properties=new_annotation_properties_obj)

        self.geom_dict2 = {
                        "type": "Point",
                        "coordinates": [1, 1]
                     }
        self.geometry_obj2 = GeometryObject.from_dict(self.geom_dict2)
        
        self.annotation_collection_obj_multifeature = AnnotationCollection(features=[self.annotation_feature_obj, self.annotation_feature_obj2])
    
    def test_annotationcollection_default_initialization(self):
        obj = AnnotationCollection()

        self.assertEqual(obj.features, None)
    
    def test_annotationcollection_initialization_with_feature(self):
        obj = AnnotationCollection(features=[self.annotation_feature_obj])

        self.assertIsInstance(obj.features, list)
        self.assertEqual(len(obj.features), 1)
        self.assertIsInstance(obj.features[0], AnnotationFeature)

    def test_annotationcollection_features_setter(self):
        obj = AnnotationFeature()

        obj.features = [self.annotation_feature_obj2]

        self.assertEqual(obj.features[0].geometry.coordinates.tolist(), [1,1])

    def test_annotationcollection_features_setter_none_features(self):
        obj = AnnotationCollection()
        obj.features = None

        self.assertIsNone(obj._features)
        self.assertIsNone(obj._feature_dict)

    def test_annotationcollection_features_setter_type_error(self):
        obj = self.annotation_collection_obj

        with pytest.raises(TypeError, match = re.escape("features must be a list of AnnotationFeatures. Got <class 'str'>")):
            obj.features = "abc"

    def test_annotationcollection_add_features(self):
        obj = self.annotation_collection_obj
        
        self.assertEqual(obj.features[0].geometry.coordinates.tolist(), [0, 0])

        obj.add_feature(self.annotation_feature_obj2)

        self.assertIsInstance(obj.features, list)
        self.assertEqual(len(obj.features), 2)
        self.assertEqual(obj.features[1].geometry.coordinates.tolist(), [1, 1])
    
    def test_annotationcollection_add_features_dict(self):
        obj = self.annotation_collection_obj

        self.assertEqual(obj.features[0].geometry.coordinates.tolist(), [0, 0])

        features_dict = {
            "type": "AnnotationFeature",
            "geometry": {
                        "type": "Point",
                        "coordinates": [1, 1]
                     },
            "properties": {
                "type": "AnnotationProperties",
                "name": "annotation3",
                "description": "new description",
                "directory": "new path"
            }
        }

        obj.add_feature(features_dict)

        self.assertIsInstance(obj.features, list)
        self.assertEqual(len(obj.features), 2)
        self.assertEqual(obj.features[1].geometry.coordinates.tolist(), [1, 1])

    def test_annotationcollection_add_invalid_feature(self):
        obj = self.annotation_collection_obj

        with pytest.raises(TypeError, match = re.escape("This requires an AnnotationFeature instance, got <class 'str'>")):
            obj.add_feature("abcd")

    def test_annotationcollection_getitem_stopiteration(self):
        obj = AnnotationCollection()
        obj.features = None

        with pytest.raises(StopIteration):
            obj.__getitem__(0)

    def test_annotation_collection_getitem_by_index(self):
        obj = self.annotation_collection_obj_multifeature

        item = obj.__getitem__(1)
        self.assertEqual(item.geometry.coordinates.tolist(), [1, 1])
    
    def test_annotation_collection_from_dict(self):
        dict = {
            "type": "AnnotationCollection",
            "features": [
                            {
                                "type": "AnnotationFeature",
                                "geometry": {
                                            "type": "Point",
                                            "coordinates": [1, 1]
                                        },
                                "properties": {
                                    "type": "AnnotationProperties",
                                    "name": "annotation3",
                                    "description": "new description",
                                    "directory": "new path"
                                }
                            }
                        ]
        }

        obj = AnnotationCollection.from_dict(dict)

        self.assertIsInstance(obj, AnnotationCollection)
        self.assertEqual(len(obj.features), 1)
        self.assertEqual(obj.features[0].geometry.coordinates.tolist(), [1, 1])

    def test_annotation_collection_from_dict_type_error(self):
        with pytest.raises(TypeError, match = re.escape("This requires a dict. Got type <class 'int'>")):
            AnnotationCollection.from_dict(1234)

    def test_annotation_collection_from_dict_key_error(self):
        dict = {"features": [
                            {
                                "type": "AnnotationFeature",
                                "geometry": {
                                            "type": "Point",
                                            "coordinates": [1, 1]
                                        },
                                "properties": {
                                    "type": "AnnotationProperties",
                                    "name": "annotation3",
                                    "description": "new description",
                                    "directory": "new path"
                                }
                            }
                        ]
                }

        with pytest.raises(KeyError, match = re.escape("the json requires the field 'type'")):
            AnnotationCollection.from_dict(dict)
        
    def test_annotation_collection_from_dict_none_features(self):
        dict = {
            "type": "AnnotationCollection",
            "features": None
        }
         
        obj = AnnotationCollection.from_dict(dict)
        self.assertIsNone(obj.features)
    
    def test_annotation_collection_from_dict_value_error(self):
        dict = {
            "type": "incorrect_type",
            "features": []
        }

        with pytest.raises(ValueError, match = re.escape("AnnotationCollection cannot be constructed from incorrect_type, expecting AnnotationCollection")):
            AnnotationCollection.from_dict(dict)

class test_fileannotationcollection(unittest.TestCase):
    def setUp(self):
        self.geometryproperties_obj = GeometryProperties(uid="abcd", name="efgh", color="blue")

        self.test_parameter_point = Point((0,0))

        self.annotation_properties_obj = AnnotationProperties(
            name = "annotation1",
            description = "abcd",
            directory = "path/folder",
            geometry_properties = [self.geometryproperties_obj],
            parameters = self.test_parameter_point
        )

        self.geom_dict = {
                        "type": "Point",
                        "coordinates": [0, 0]
                     }
        self.geometry_obj = GeometryObject.from_dict(self.geom_dict)
    
        self.annotation_feature_obj = AnnotationFeature(geometry=self.geometry_obj, properties=self.annotation_properties_obj)

        self.annotation_collection_obj =  AnnotationCollection(features=[self.annotation_feature_obj])
        
        self.annotation_collection = AnnotationCollection(features=[self.annotation_feature_obj])

        version = "test_version"
        annotations = self.annotation_collection_obj
        image_file_name = "test_image_file_name"
        image_id = "test_image_id"
        core_name = "test_core_name"

        self.file_annotation_collection_obj = FileAnnotationCollection(version=version, annotations=annotations, image_file_name=image_file_name, image_id=image_id, core_name=core_name)
    
    def test_fileannotationcollection_initialization(self):
        _version = "test_version"
        _annotations = self.annotation_collection_obj
        _image_file_name = "test_image_file_name"
        _image_id = "test_image_id"
        _core_name = "test_core_name"

        obj = FileAnnotationCollection(version=_version, annotations=_annotations, image_file_name=_image_file_name, image_id=_image_id, core_name=_core_name)
        
        self.assertEqual(obj.version, _version)
        self.assertEqual(obj.annotations, _annotations)
        self.assertEqual(obj.image_file_name, _image_file_name)
        self.assertEqual(obj.image_id, _image_id)
        self.assertEqual(obj.core_name, _core_name)
        
    def test_fileannotationcollection_none_initialization(self):
        with self.assertLogs('sarpy.annotation.base', level='ERROR') as context_manager:
            obj = FileAnnotationCollection(image_file_name=None, image_id=None, core_name=None)

        # check that the expected error message is in the logs
        self.assertTrue(
            any(
                "One of image_file_name, image_id, or core_name should be defined"
                in msg for msg in context_manager.output
            ),
            "Expected error log not found"
        )
        
        self.assertEqual(obj.version, "Base:1.0")    
        self.assertIsNone(obj._annotations)
        self.assertIsNone(obj._image_file_name)
        self.assertIsNone(obj._image_id)
        self.assertIsNone(obj._core_name)

    def test_fileannotationcollection_initialization_type_error(self):
        _version = "test_version"
        _annotations = self.annotation_collection_obj
        _image_file_name = 1234
        _image_id = "test_image_id"
        _core_name = "test_core_name"

        with pytest.raises(TypeError, match = re.escape("image_file_name must be a None or a string")):
            obj = FileAnnotationCollection(version=_version, annotations=_annotations, image_file_name=_image_file_name, image_id=_image_id, core_name=_core_name)
    
    def test_fileannotationcollection_version_property(self):
        obj = self.file_annotation_collection_obj
        
        self.assertEqual(obj.version, "test_version")

    def test_fileannotationcollection_image_file_name_property(self):
        obj = self.file_annotation_collection_obj
        
        self.assertEqual(obj.image_file_name, "test_image_file_name")

    def test_fileannotationcollection_image_id_property(self):
        obj = self.file_annotation_collection_obj
        
        self.assertEqual(obj.image_id, "test_image_id")
    
    def test_fileannotationcollection_core_name_property(self):
        obj = self.file_annotation_collection_obj
        
        self.assertEqual(obj.core_name, "test_core_name")
    
    def test_fileannotationcollection_annotations_property(self):
        obj = self.file_annotation_collection_obj
        annotations = obj.annotations

        self.assertEqual(annotations.features[0].geometry.coordinates.tolist(), [0, 0])
        self.assertIsInstance(annotations, AnnotationCollection)
    
    def test_fileannotationcollection_annotations_setter_none(self):
        obj = self.file_annotation_collection_obj

        obj.annotations = None

        self.assertIsNone(obj.annotations)
    
    def test_fileannotationcollection_annotations_setter_annotation_collection(self):
        obj = self.file_annotation_collection_obj

        obj.annotations = self.annotation_collection

        self.assertEqual(obj.annotations.features[0].geometry.coordinates.tolist(), [0, 0])
        self.assertIsInstance(obj.annotations, AnnotationCollection)
    
    def test_fileannotationcollection_annotations_setter_dict(self):
        dict = {
            "type": "AnnotationCollection",
            "features": [
                            {
                                "type": "AnnotationFeature",
                                "geometry": {
                                            "type": "Point",
                                            "coordinates": [1, 1]
                                        },
                                "properties": {
                                    "type": "AnnotationProperties",
                                    "name": "annotation3",
                                    "description": "new description",
                                    "directory": "new path"
                                }
                            }
                        ]
        }
        
        obj = self.file_annotation_collection_obj
        obj.annotations = dict

        self.assertEqual(obj.annotations.features[0].geometry.coordinates.tolist(), [1, 1])
        self.assertIsInstance(obj.annotations, AnnotationCollection)
    
    def test_fileannotationcollection_annotations_setter_type_error(self):
        obj = self.file_annotation_collection_obj

        with pytest.raises(TypeError, match = re.escape("annotations must be an AnnotationCollection. Got type <class 'int'>")):
            obj.annotations = 1234
    
    def test_fileannotationcollection_add_annotation_dict(self):
        features_dict = {
            "type": "AnnotationFeature",
            "geometry": {
                        "type": "Point",
                        "coordinates": [1, 1]
                     },
            "properties": {
                "type": "AnnotationProperties",
                "name": "annotation3",
                "description": "new description",
                "directory": "new path"
            }
        }

        obj = self.file_annotation_collection_obj

        self.assertEqual(obj.annotations.features[0].geometry.coordinates.tolist(), [0, 0])
        self.assertEqual(len(obj.annotations), 1)

        obj.add_annotation(features_dict)

        self.assertEqual(len(obj.annotations), 2)
        self.assertEqual(obj.annotations.features[1].geometry.coordinates.tolist(), [1, 1])
    
    def test_fileannotationcollection_add_annotation_type_error(self):
        obj = self.file_annotation_collection_obj

        with pytest.raises(TypeError, match = re.escape("This requires an AnnotationFeature instance. Got <class 'int'>")):
            obj.add_annotation(1234)
    
    def test_fileannotationcollection_add_annotation_to_none(self):
        obj = FileAnnotationCollection()

        obj.add_annotation(self.annotation_feature_obj)

        self.assertIsInstance(obj._annotations, AnnotationCollection)
        self.assertEqual(obj.annotations[0].geometry.coordinates.tolist(), [0, 0])
    
    def test_fileannotationcollection_add_annotation_annotationfeature(self):
        obj = self.file_annotation_collection_obj

        self.assertEqual(obj.annotations.features[0].geometry.coordinates.tolist(), [0, 0])
        self.assertEqual(len(obj.annotations), 1)

        obj.add_annotation(self.annotation_feature_obj)

        self.assertEqual(len(obj.annotations), 2)
        self.assertEqual(obj.annotations[1].geometry.coordinates.tolist(), [0, 0])
    
    def test_fileannotationcollection_delete_annotation(self):
        obj = self.file_annotation_collection_obj

        # confirms the number of annotations is 1
        self.assertEqual(len(obj.annotations), 1)

        # adds an annotation so that there are 2 total
        obj.add_annotation(self.annotation_feature_obj)

        # confirms that there are 2 annotations
        self.assertEqual(len(obj.annotations), 2)

        # gets the annotation id for the added object
        annotation_id = self.annotation_feature_obj.uid

        # deletes the annotation by id
        obj.delete_annotation(annotation_id) # find out what i should be using for annotation id

        # confirms that the annotation is successfully deleted
        self.assertEqual(len(obj.annotations), 1)

    def test_fileannotationcollection_from_file(self):
        obj = FileAnnotationCollection.from_file("tests/annotation/fileannotationcollection_from_file_test.json")

        self.assertIsInstance(obj, FileAnnotationCollection)
        self.assertEqual(obj.version, "test_version")
        self.assertEqual(obj.image_file_name, "test_image_file_name")
        self.assertEqual(obj.image_id, "test_image_id")
        self.assertEqual(obj.core_name, "test_core_name")

    def test_fileannotationcollection_from_dict_non_dict(self):
        with pytest.raises(TypeError, match = re.escape("This requires a dict. Got type <class 'int'>")):
            obj = FileAnnotationCollection.from_dict(1234)
    
    def test_fileannotationcollection_from_dict_value_error(self):
        dict = {
            "type": "incorrect_type"
        }

        with pytest.raises(ValueError, match = re.escape('FileAnnotationCollection cannot be constructed from incorrect_type, expecting FileAnnotationCollection')):
            obj = FileAnnotationCollection.from_dict(dict)
    
    def test_fileannotationcollection_from_dict(self):
        dict = {
                "type": "FileAnnotationCollection",
                "version": "test_version",
                "image_file_name": "test_image_file_name",
                "image_id": "test_image_id",
                "core_name": "test_core_name",
                "annotations": {
                    "type": "AnnotationCollection",
                    "features": [
                            {
                                "type": "AnnotationFeature",
                                "id": "10dfc96e-ed21-4214-99bf-6543f079acff",
                                "geometry": {
                                    "type": "Point",
                                    "coordinates": [
                                            0.0,
                                            0.0
                                        ]
                                    },
                                "properties": {
                                "type": "AnnotationProperties",
                                "name": "annotation1",
                                "description": "abcd",
                                "directory": "path/folder",
                                "geometry_properties": [
                                        {
                                            "type": "GeometryProperties",
                                            "uid": "abcd",
                                            "name": "efgh",
                                            "color": "blue"
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                }
        
        obj = FileAnnotationCollection.from_dict(dict)
       
        self.assertIsInstance(obj, FileAnnotationCollection)
        self.assertEqual(obj.version, "test_version")
        self.assertEqual(obj.image_file_name, "test_image_file_name")
        self.assertEqual(obj.image_id, "test_image_id")
        self.assertEqual(obj.core_name, "test_core_name")

    def test_fileannotationcollection_to_dict_none(self):
        obj = FileAnnotationCollection()

        return_dict = obj.to_dict()

        self.assertEqual(return_dict.get("type"), "FileAnnotationCollection")
        self.assertEqual(return_dict.get("version"), "Base:1.0")   
    
    def test_fileannotationcollection_to_dict(self):
        obj = self.file_annotation_collection_obj

        return_dict = obj.to_dict()

        self.assertEqual(return_dict.get("type"), "FileAnnotationCollection")
        self.assertEqual(return_dict.get("image_file_name"), "test_image_file_name")
        self.assertEqual(return_dict.get("version"), "test_version"),
        self.assertEqual(return_dict.get("image_id"), "test_image_id")
        self.assertEqual(return_dict.get("core_name"), "test_core_name")

        test_annotation = self.annotation_collection_obj.to_dict()
        self.assertEqual(return_dict.get("annotations"), test_annotation)
    
    def test_fileannoationcollection_to_dict_with_parent_dict(self):
        parent_dict = {
            "type": "FileAnnotationCollection",
            "image_file_name": "un-updated image_file_name",
            "version": "un-updated version",
            "image_id": "un-updated image_id",
            "core_name": "un-updated core_name"
        }
        
        self.file_annotation_collection_obj.to_dict(parent_dict)

        # fields should be updated from the file_annotation_collection_obj
        self.assertEqual(parent_dict.get("type"), "FileAnnotationCollection")
        self.assertEqual(parent_dict.get("image_file_name"), "test_image_file_name")
        self.assertEqual(parent_dict.get("version"), "test_version"),
        self.assertEqual(parent_dict.get("image_id"), "test_image_id")
        self.assertEqual(parent_dict.get("core_name"), "test_core_name")

        test_annotation = self.annotation_collection_obj.to_dict()
        self.assertEqual(parent_dict.get("annotations"), test_annotation)
    
    def test_fileannotationcollection_to_file(self):
        obj = self.file_annotation_collection_obj

        obj.to_file("tests/annotation/test_fileannotationcollection_to_file_function.json")
    