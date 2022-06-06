"""
Functionality for an aggregate sicd type reader, for opening multiple sicd type
files as a single reader object.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


from typing import Tuple, Sequence, Union

from sarpy.io.complex.converter import open_complex
from sarpy.io.general.base import AggregateReader, SarpyIOError
from sarpy.io.complex.base import SICDTypeReader
from sarpy.io.complex.sicd_elements.SICD import SICDType


class AggregateComplexReader(AggregateReader, SICDTypeReader):
    """
    Aggregate multiple sicd type readers into a single reader instance.

    **Changed in version 1.3.0** for reading changes.
    """

    __slots__ = ('_readers', '_index_mapping')

    def __init__(self, readers: Union[Sequence[str], Sequence[SICDTypeReader]]):
        """

        Parameters
        ----------
        readers : Sequence[str]|Sequence[SICDTypeReader]
        """

        readers = self._validate_readers(readers)
        AggregateReader.__init__(self, readers)
        sicds = self._define_sicds()
        SICDTypeReader.__init__(self, None, sicds)
        self._check_sizes()

    @staticmethod
    def _validate_readers(readers : Sequence[SICDTypeReader]) -> Tuple[SICDTypeReader, ...]:
        """
        Validate the input reader/file collection.

        Parameters
        ----------
        readers : Sequence[SICDTypeReader]

        Returns
        -------
        Tuple[SICDTypeReader]
        """

        if not isinstance(readers, Sequence):
            raise TypeError('input argument must be a list or tuple of readers/files. Got type {}'.format(type(readers)))

        # get a reader for each entry, and make sure that they are sicd type

        # validate each entry
        the_readers = []
        for i, entry in enumerate(readers):
            if isinstance(entry, str):
                try:
                    reader = open_complex(entry)
                except SarpyIOError:
                    raise SarpyIOError(
                        'Attempted and failed to open {} (entry {} of the input argument) '
                        'using the complex opener.'.format(entry, i))
            else:
                reader = entry

            if not isinstance(reader, SICDTypeReader):
                raise ValueError(
                    'Entry {} of the input argument does not correspond to a SICDTypeReader instance. '
                    'Got type {}.'.format(i, type(reader)))
            the_readers.append(reader)
        return tuple(the_readers)

    def _define_sicds(self)-> Tuple[SICDType, ...]:
        sicds = []
        for reader_index, sicd_index in self.index_mapping:
            reader = self._readers[reader_index]
            sicd = reader.get_sicds_as_tuple()[sicd_index]
            sicds.append(sicd)
        return tuple(sicds)
