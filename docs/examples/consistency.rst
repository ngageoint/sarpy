Standards Consistency Checking
==============================

The subpackage `sarpy.consistency` provide a collection of command-line utilities
for checking the consistency of SICD, SIDD, and CPHD files against the given
standard.

SICD Validation
---------------

The `sarpy.consistency.sicd_consistency` module provides a utility for
checking the validity of a SICD file. For scripting usage, do

.. code-block:: python

    from sarpy.consistency.sicd_constency import check_file
    check_file('<path to sicd file>')

Alternatively, from the command line perform

>>> python -m sarpy.consistency.sicd_consistency <path to sicd file>

For more information, about command line usage, see

>>> python -m sarpy.consistency.sicd_consistency --help

SIDD Validation
---------------

The `sarpy.consistency.sidd_consistency` module provides a utility for
checking the validity of a SIDD file. For scripting usage, do

.. code-block:: python

    from sarpy.consistency.sidd_constency import check_file
    check_file('<path to sidd file>')

Alternatively, from the command line perform

>>> python -m sarpy.consistency.sidd_consistency <path to sidd file>

For more information, about command line usage, see

>>> python -m sarpy.consistency.sidd_consistency --help

CPHD Validation
---------------

The `sarpy.consistency.cphd_consistency` module provides a utility for
checking the validity of a CPHD file. For scripting usage, do

.. code-block:: python

    from sarpy.consistency.cphd_constency import check_file
    check_file('<path to cphd file>')

Alternatively, from the command line perform

>>> python -m sarpy.consistency.cphd_consistency <path to cphd file>

For more information, about command line usage, see

>>> python -m sarpy.consistency.cphd_consistency --help
