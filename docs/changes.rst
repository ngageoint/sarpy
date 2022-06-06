**********************************
Important changes in version 1.3.0
**********************************

Subpackage restructuring
------------------------
- Most elements of the `sarpy.processing` subpackage have been moved to newly
  created subpackage `sarpy.processing.sicd`.
- The modules `sidd_product_creation` and `sidd_structure_creation` have been
  moved from the subpackage `sarpy.io.product` to newly created subpackage
  `sarpy.processing.sidd`.
- The module `kmz_product_creation` has been moved from the subpackage
  `sarpy.io.product` to the subpackage `sarpy.visualization`.

Reading changes
---------------
For SICD-like, CPHD, and CRSD readers in sarpy take slicing arguments, read image
data of some native format, and perform some reformating operation to return a
two-dimensional image of complex64 data type. Although expected to be the
minority (by far) use case, the ability to access raw data may be useful, and
has been introduced.

.. code:: python

    # assuming reader defined above
    <raw data> = reader[<row slice>, <col slice>, <image index>, 'raw']
    # or
    <raw data> = reader.read_raw(*slicing, index=<image index>)
    # or
    <raw data> = reader(*slicing, raw=True, index=<image index>)

This is most non-trivial to CPHD and CRSD files with a `AmpSF` PVP present.

It should be noted that the slicing in raw format reading applies to the native
slicing, so any symmetry operations used to reformat the raw data to the formatted
output (i.e. transpose of rows and columns or reversal of rows and/or columns)
will not have been performed. These operations are not used in actual SICD, CPHD,
and CRSD files, but **likely are** used in SLC format data which is read by the
appropriate SICD-type reader.

Writing changes
---------------
Similarly, writers for SICD, CPHD, and CRSD format files take a two-dimensional
of complex64 data type and something like a slicing argument, and perform some
reformating operation to write image data in some native format. The ability to
provide and directly write raw data (avoiding the reformatting step) has been
introduced.

.. code:: python

    # assuming writer defined above
    writer.write_raw(
        <raw data>, start_indices=<start_indices>, subscript=<slicing definition>,
        index=<image index>)
    # or
    writer(
        <raw data>, start_indices=<start_indices>, subscript=<slicing definition>,
        index=<image index>, raw=True)

Look to the documentation for explanation of `start_indices` and `subscript`.

**This change is most important to CPHD and CRSD files.** Prior to this change,
writing a CPHD or CRSD file with a scale factor was asymmetric from reading the
same file. That is, the image data provided by the reader had the `AmpSF`
included in the formatting operation, while the writer did not account for the
presence of the `AmpSF`. This has been corrected.

When writing a CPHD or CRSD file, the complex data will account for the presence
of the `AmpSF`. That is, the complex data will be divided by the appropriate
scale factor array derived from the PVP data. In light of this, the corresponding
PVP data must be written before the signal array when `AmpSF` is defined.
