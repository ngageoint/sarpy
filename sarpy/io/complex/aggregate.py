# -*- coding: utf-8 -*-
"""
Functionality for an aggregate sicd type reader, for opening multiple sicd type
files as a single reader object.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


from typing import List, Tuple

from sarpy.compliance import string_types
from sarpy.io.complex.converter import open_complex
from sarpy.io.general.base import BaseReader, AggregateReader


class AggregateComplexReader(AggregateReader):
    """
    Aggregate multiple sicd type files and/or readers into a single reader instance.
    """

    __slots__ = ('_readers', '_index_mapping')

    def __init__(self, readers):
        """

        Parameters
        ----------
        readers : List[BaseReader|str]
        """

        readers = self._validate_readers(readers)
        super(AggregateComplexReader, self).__init__(readers, reader_type="SICD")
        # define the SICD associated structures
        self._define_sicds()

    @staticmethod
    def _validate_readers(readers):
        """
        Validate the input reader/file collection.

        Parameters
        ----------
        readers : list|tuple

        Returns
        -------
        Tuple[BaseReader]
        """

        if not isinstance(readers, (list, tuple)):
            raise TypeError('input argument must be a list or tuple of readers/files. Got type {}'.format(type(readers)))

        # get a reader for each entry, and make sure that they are sicd type

        # validate each entry
        the_readers = []
        for i, entry in enumerate(readers):
            if isinstance(entry, string_types):
                try:
                    reader = open_complex(entry)
                except IOError:
                    raise IOError(
                        'Attempted and failed to open {} (entry {} of the input argument) '
                        'using the complex opener.'.format(entry, i))
            elif not isinstance(entry, BaseReader):
                raise TypeError(
                    'All elements of the input argument must be file names or BaseReader instances. '
                    'Entry {} is of type {}'.format(i, type(entry)))
            else:
                reader = entry

            if reader.reader_type != "SICD":
                raise ValueError(
                    'Entry {} of the input argument does not correspond to a sicd type reader. '
                    'Got type {}, with reader_type value'.format(i, type(reader), reader.reader_type))
            the_readers.append(reader)
        return tuple(the_readers)

    def _define_sicds(self):
        sicds = []
        for reader_index, sicd_index in self.index_mapping:
            reader = self._readers[reader_index]
            sicd = reader.get_sicds_as_tuple()[sicd_index]
            sicds.append(sicd)
        self._sicd_meta = tuple(sicds)
