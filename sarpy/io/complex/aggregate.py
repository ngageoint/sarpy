# -*- coding: utf-8 -*-
"""
Functionality for an aggregate reader, for opening multiple files as a single
reader object.
"""

from typing import List, Tuple

from sarpy.compliance import string_types
from .converter import open_complex
from ..general.base import BaseReader


class AggregateReader(BaseReader):
    """
    Aggregate multiple files and/or readers into a single reader instance.
    """

    __slots__ = ('_readers', '_index_mapping')

    def __init__(self, readers):
        """

        Parameters
        ----------
        readers : List[BaseReader|str]
        """

        self._index_mapping = None
        self._readers = self._validate_readers(readers)
        the_sicds, the_chippers = self._define_index_mapping()
        super(AggregateReader, self).__init__(sicd_meta=the_sicds, chipper=the_chippers, is_sicd_type=True)

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
            if not reader.is_sicd_type:
                raise ValueError(
                    'Entry {} of the input argument does not correspond to a sicd type reader. '
                    'Got type {}'.format(i, type(reader)))
            the_readers.append(reader)
        return tuple(the_readers)

    def _define_index_mapping(self):
        """
        Define the index mapping.

        Returns
        -------
        Tuple[SICDType], Tuple[BaseChipper]
        """

        # prepare the index mapping workspace
        index_mapping = []
        # assemble the sicd_meta and chipper arguments
        the_sicds = []
        the_chippers = []
        for i, reader in enumerate(self._readers):
            for j, (sicd, chipper) in enumerate(zip(reader.get_sicds_as_tuple(), reader._get_chippers_as_tuple())):
                the_sicds.append(sicd)
                the_chippers.append(chipper)
                index_mapping.append((i, j))

        self._index_mapping = tuple(index_mapping)
        return tuple(the_sicds), tuple(the_chippers)

    @property
    def index_mapping(self):
        # type: () -> Tuple[Tuple[int, int]]
        """
        Tuple[Tuple[int, int]]: The index mapping of the form (reader index, sicd index).
        """

        return self._index_mapping

    @property
    def file_name(self):
        # type: () -> Tuple[str]
        """
        Tuple[str]: The filename collection.
        """

        return tuple(entry.file_name for entry in self._readers)
