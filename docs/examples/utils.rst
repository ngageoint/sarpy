Command-line Utilities
======================

Description of the command-line utilities which help accomplish a variety of tasks.


Convert to SICD format
----------------------

This can be accomplished using a command-line utility as

>>> python -m sarpy.utils.convert_to_sicd <input file> <output directory>

For a basic help, check

>>> python -m sarpy.utils.convert_to_sicd --help


Chip SICD
---------
This can be accomplished using a command-line utility as

>>> python -m sarpy.utils.chip_sicd <input file> <output directory>

For a basic help, check

>>> python -m sarpy.utils.chip_sicd --help


Derived Product Creation from SICD
----------------------------------

This can be accomplished using a command-line utility as

>>> python -m sarpy.utils.create_product <input file> <output directory>

For a basic help, check

>>> python -m sarpy.utils.create_product --help

KMZ Product Creation from SICD
------------------------------

This can be accomplished using a command-line utility as

>>> python -m sarpy.utils.create_kmz <input file> <output directory> -v

For a basic help on the command-line, check

>>> python -m sarpy.utils.create_kmz --help


Basic CPHD Metadata Dump
------------------------

To dump the CPHD header and metadata structure from the command-line

>>> python -m sarpy.utils.cphd_utils <path to cphd file>

For a basic help on the command-line, check

>>> python -m sarpy.utils.cphd_utils --help


Full NITF Header Dump
---------------------

To dump NITF header information to a text file from the command-line

>>> python -m sarpy.utils.nitf_utils <path to nitf file>

For a basic help on the command-line, check

>>> python -m sarpy.utils.nitf_utils --help


Add Nominal Noise Polynomial
----------------------------

This can be accomplished using a command-line utility as

>>> python -m sarpy.utils.nominal_sicd_noise <input file> <output directory>

For a basic help, check

>>> python -m sarpy.utils.nominal_sicd_noise --help

