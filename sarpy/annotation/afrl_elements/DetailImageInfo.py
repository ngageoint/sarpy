"""
Definition for the DetailImageInfo AFRL labeling object
"""

__classification__ = "UNCLASSIFIED"
__authors__ = ("Thomas McCullough", "Thomas Rackers")

from typing import Optional

# noinspection PyProtectedMember
from sarpy.io.complex.sicd_elements.base import _StringDescriptor, Serializable


class ClassificationMarkingsType(Serializable):
    _fields = (
        'Classification','Restrictions', 'ClassifiedBy', 'DeclassifyOn', 'DerivedFrom')
    _required = ('Classification','Restrictions')
    # descriptors
    Classification = _StringDescriptor(
        'Classification', _required, default_value='')  # type: str
    Restrictions = _StringDescriptor(
        'Restrictions', _required, default_value='')  # type: str
    ClassifiedBy = _StringDescriptor(
        'ClassifiedBy', _required)  # type: Optional[str]
    DeclassifyOn = _StringDescriptor(
        'DeclassifyOn', _required)  # type: Optional[str]
    DerivedFrom = _StringDescriptor(
        'DerivedFrom', _required)  # type: Optional[str]

    def __init__(self, Classification='',Restrictions='', ClassifiedBy=None,
                 DeclassifyOn=None, DerivedFrom=None, **kwargs):
        """
        Parameters
        ----------
        Classification : str
        Restrictions : str
        ClassifiedBy : None|str
        DeclassifyOn : None|str
        DerivedFrom : None|str
        kwargs
            Other keyword arguments
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Classification = Classification
        self.Restrictions = Restrictions
        self.ClassifiedBy = ClassifiedBy
        self.DeclassifyOn = DeclassifyOn
        self.DerivedFrom = DerivedFrom
        super(ClassificationMarkingsType, self).__init__(**kwargs)


# TODO: to be completed

class DetailImageInfoType(Serializable):
    _fields = ()  # fill this in
    _required = ()  # fill this in

    # descriptors

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        kwargs
            Other keyword arguments
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        super(DetailImageInfoType, self).__init__(**kwargs)

