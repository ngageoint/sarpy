# -*- coding: utf-8 -*-
"""
The ProductProcessingType definition.
"""

from collections import OrderedDict
from xml.etree import ElementTree
from typing import List

from .base import DEFAULT_STRICT

# noinspection PyProtectedMember
from ..sicd_elements.base import Serializable, _ParametersDescriptor, ParametersCollection, \
    _StringDescriptor, _find_children

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class ProcessingModuleType(Serializable):
    """
    Flexibly structured processing module definition to keep track of the name and any parameters associated
    with the algorithms used to produce the SIDD.
    """

    _fields = ('ModuleName', 'name', 'ModuleParameters')
    _required = ('ModuleName', 'name', 'ModuleParameters')
    _set_as_attribute = ('name', )
    _collections_tags = {'ModuleParameters': {'array': False, 'child_tag': 'ModuleParameter'}}
    # Descriptor
    ModuleName = _StringDescriptor(
        'ModuleName', _required, strict=DEFAULT_STRICT,
        docstring='The module name.')  # type: str
    name = _StringDescriptor(
        'name', _required, strict=DEFAULT_STRICT,
        docstring='The module identifier.')  # type: str
    ModuleParameters = _ParametersDescriptor(
        'ModuleParameters', _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: ParametersCollection

    def __init__(self, ModuleName=None, name=None, ModuleParameters=None, ProcessingModules=None, **kwargs):
        """

        Parameters
        ----------
        ModuleName : str
        name : str
        ModuleParameters : None|ParametersCollection|dict
        ProcessingModules : None|List[ProcessingModuleType]
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.ModuleName = ModuleName
        self.name = name
        self.ModuleParameters = ModuleParameters

        self._ProcessingModules = []
        if ProcessingModules is None:
            pass
        elif isinstance(ProcessingModules, ProcessingModuleType):
            self.addProcessingModule(ProcessingModules)
        elif isinstance(ProcessingModules, (list, tuple)):
            for el in ProcessingModules:
                self.addProcessingModule(el)
        else:
            raise ('ProcessingModules got unexpected type {}'.format(type(ProcessingModules)))
        super(ProcessingModuleType, self).__init__(**kwargs)

    @property
    def ProcessingModules(self):
        """List[ProcessingModuleType]: list of ProcessingModules."""
        return self._ProcessingModules

    def getProcessingModule(self, key):
        """
        Get ProcessingModule(s) with name attribute == `key`.

        Parameters
        ----------
        key : str

        Returns
        -------
        List[ProcessingModuleType]
        """

        return [entry for entry in self._ProcessingModules if entry.name == key]

    def addProcessingModule(self, value):
        """
        Add the ProcessingModule to the list.

        Parameters
        ----------
        value : ProcessingModuleType

        Returns
        -------
        None
        """

        if isinstance(value, ElementTree.Element):
            value = ProcessingModuleType.from_node(value, self._xml_ns, ns_key=self._xml_ns_key)
        elif isinstance(value, dict):
            value = ProcessingModuleType.from_dict(value)

        if isinstance(value, ProcessingModuleType):
            self._ProcessingModules.append(value)
        else:
            raise TypeError('Trying to set ProcessingModule with unexpected type {}'.format(type(value)))

    @classmethod
    def from_node(cls, node, xml_ns, ns_key=None, kwargs=None):
        if kwargs is None:
            kwargs = OrderedDict()
        kwargs['ProcessingModules'] = _find_children(node, 'ProcessingModule', xml_ns, ns_key)
        return super(ProcessingModuleType, cls).from_node(node, xml_ns, ns_key=ns_key, kwargs=kwargs)

    def to_node(self, doc, tag, ns_key=None, parent=None, check_validity=False, strict=DEFAULT_STRICT, exclude=()):
        node = super(ProcessingModuleType, self).to_node(
            doc, tag, ns_key=ns_key, parent=parent, check_validity=check_validity, strict=strict, exclude=exclude)
        # slap on the ProcessingModule children
        for entry in self._ProcessingModules:
            entry.to_node(doc, tag, ns_key=ns_key, parent=node, strict=strict)
        return node

    def to_dict(self, check_validity=False, strict=DEFAULT_STRICT, exclude=()):
        out = super(ProcessingModuleType, self).to_dict(check_validity=check_validity, strict=strict, exclude=exclude)
        # slap on the GeoInfo children
        if len(self.ProcessingModules) > 0:
            out['ProcessingModules'] = [
                entry.to_dict(check_validity=check_validity, strict=strict) for entry in self._ProcessingModules]
        return out


class ProductProcessingType(Serializable):
    """
    Computed metadata regarding one or more of the input collections and final product.
    """

    _fields = ()
    _required = ()

    def __init__(self, ProcessingModules=None, **kwargs):
        """

        Parameters
        ----------
        ProcessingModules : None|List[ProcessingModuleType]
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']

        self._ProcessingModules = []
        if ProcessingModules is None:
            pass
        elif isinstance(ProcessingModules, ProcessingModuleType):
            self.addProcessingModule(ProcessingModules)
        elif isinstance(ProcessingModules, (list, tuple)):
            for el in ProcessingModules:
                self.addProcessingModule(el)
        else:
            raise ('ProcessingModules got unexpected type {}'.format(type(ProcessingModules)))
        super(ProductProcessingType, self).__init__(**kwargs)

    @property
    def ProcessingModules(self):
        """List[ProcessingModuleType]: list of ProcessingModules."""
        return self._ProcessingModules

    def getProcessingModule(self, key):
        """
        Get ProcessingModule(s) with name attribute == `key`.

        Parameters
        ----------
        key : str

        Returns
        -------
        List[ProcessingModuleType]
        """

        return [entry for entry in self._ProcessingModules if entry.name == key]

    def addProcessingModule(self, value):
        """
        Add the ProcessingModule to the list.

        Parameters
        ----------
        value : ProcessingModuleType

        Returns
        -------
        None
        """

        if isinstance(value, ElementTree.Element):
            value = ProcessingModuleType.from_node(value, self._xml_ns, ns_key=self._xml_ns_key)
        elif isinstance(value, dict):
            value = ProcessingModuleType.from_dict(value)

        if isinstance(value, ProcessingModuleType):
            self._ProcessingModules.append(value)
        else:
            raise TypeError('Trying to set ProcessingModule with unexpected type {}'.format(type(value)))

    @classmethod
    def from_node(cls, node, xml_ns, ns_key=None, kwargs=None):
        if kwargs is None:
            kwargs = OrderedDict()
        kwargs['ProcessingModules'] = _find_children(node, 'ProcessingModule', xml_ns, ns_key)
        return super(ProductProcessingType, cls).from_node(node, xml_ns, ns_key=ns_key, kwargs=kwargs)

    def to_node(self, doc, tag, ns_key=None, parent=None, check_validity=False, strict=DEFAULT_STRICT, exclude=()):
        node = super(ProductProcessingType, self).to_node(
            doc, tag, ns_key=ns_key, parent=parent, check_validity=check_validity, strict=strict, exclude=exclude)
        # slap on the ProcessingModule children
        for entry in self._ProcessingModules:
            entry.to_node(doc, tag, ns_key=ns_key, parent=node, strict=strict)
        return node

    def to_dict(self, check_validity=False, strict=DEFAULT_STRICT, exclude=()):
        out = super(ProductProcessingType, self).to_dict(check_validity=check_validity, strict=strict, exclude=exclude)
        # slap on the GeoInfo children
        if len(self.ProcessingModules) > 0:
            out['ProcessingModules'] = [
                entry.to_dict(check_validity=check_validity, strict=strict) for entry in self._ProcessingModules]
        return out
