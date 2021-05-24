Geometry/Projection Methods
===========================

Basic methods for image geometry and associated requirements.

Point Projection Methods from SICD/SIDD
---------------------------------------

Projection methods for converting between image coordinates and "physical"
coordinates in SICD type or SIDD type images.

Here is some basic example usage:

.. code-block:: python

    from sarpy.geometry import point_projection
    from sarpy.io.complex.sicd import SICDReader

    reader = SICDReader('<path to sicd file>')
    structure = reader.sicd_meta
    # or reader.reader.get_sicds_as_tuple()[0]

    # you can also use a SIDD structure, obtained from a product type reader

    # assume that ecf_coords is some previously defined numpy array of
    # shape (..., 3), with final dimension [X, Y, Z]
    image_coords = point_projection.ground_to_image(ecf_coords, structure)
    # image_coords will be a numpy array of shape (..., 2),
    # with final dimension [row, column]

    # assume that geo_coords is some previously defined numpy array of
    # shape (..., 3), with final dimension [lat, lon, hae]
    image_coords = point_projection.ground_to_image_geo(geo_coords, structure)
    # image_coords will be a numpy array of shape (..., 2),
    # with final dimension [row, column]

    # assume that image_coords is some previously defined numpy array of
    # shape (..., 2) with final dimension [row, column]
    ecf_coords_fixed_hae = \
        point_projection.image_to_ground(image_coords, structure, projection_type='HAE')
    ecf_coords_plane = \
        point_projection.image_to_ground(image_coords, structure, projection_type='PLANE')
    geo_coords_fixed_hae = \
        point_projection.image_to_ground_geo(image_coords, structure, projection_type='HAE')
    geo_coords_plane = \
        point_projection.image_to_ground_geo(image_coords, structure, projection_type='PLANE')
    # these outputs will be numpy arrays of shape (..., 3)
    # NB: "fixed" hae will exhibit approximately, but not exactly, constant HAE value

    # alternatively, these are also methods of the sicd/sidd structure
    # it is a matter of coding preference
    image_coords = structure.project_ground_to_image(ecf_coords)
    image_coords = structure.project_ground_to_image_geo(geo_coords)
    ecf_coords_fixed_hae = structure.project_image_to_ground(image_coords, projection_type='HAE')
    ecf_coords_plane = structure.project_image_to_ground(image_coords, projection_type='PLANE')
    geo_coords_fixed_hae = structure.project_image_to_ground_geo(image_coords, projection_type='HAE')
    geo_coords_plane = structure.project_image_to_ground_geo(image_coords, projection_type='PLANE')

See :mod:`sarpy.geometry.point_projection` for more explicit documentation.
