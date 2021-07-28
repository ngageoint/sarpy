"""
Definition for the DetailSubCollectionInfo AFRL labeling object
"""

__classification__ = "UNCLASSIFIED"
__authors__ = ("Thomas McCullough", "Thomas Rackers")


from sarpy.io.complex.sicd_elements.base import Serializable

# TODO: to be completed

class DetailSubCollectionInfoType(Serializable):
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
        super(DetailSubCollectionInfoType, self).__init__(**kwargs)


