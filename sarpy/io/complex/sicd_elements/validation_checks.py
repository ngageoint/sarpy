# -*- coding: utf-8 -*-
"""
The detailed and involved validity checks for the sicd structure.

Note: These checks were originally implemented in the SICDType object,
but separating this implementation is probably less confusing in the
long run.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


def _validate_scp_time(the_sicd):
    """
    Validate the SCPTime.

    Parameters
    ----------
    the_sicd : sarpy.io.complex.sicd_elements.SICD.SICDType

    Returns
    -------
    bool
    """

    if the_sicd.SCPCOA is None or the_sicd.SCPCOA.SCPTime is None or \
            the_sicd.Grid is None or the_sicd.Grid.TimeCOAPoly is None:
        return True

    cond = True
    val1 = the_sicd.SCPCOA.SCPTime
    val2 = the_sicd.Grid.TimeCOAPoly[0, 0]
    if abs(val1 - val2) > 1e-6:
        the_sicd.log_validity_error(
            'SCPTime populated as {},\n'
            'and constant term of TimeCOAPoly populated as {}'.format(val1, val2))
        cond = False
    return cond


def _validate_image_form_parameters(the_sicd, alg_type):
    """
    Validate the image formation parameter specifics.

    Parameters
    ----------
    the_sicd : sarpy.io.complex.sicd_elements.SICD.SICDType
    alg_type : str

    Returns
    -------
    bool
    """

    cond = True
    if the_sicd.ImageFormation.ImageFormAlgo is None:
        the_sicd.log_validity_warning(
            'Image formation algorithm(s) {} populated, but ImageFormation.ImageFormAlgo was not set.\n'
            'ImageFormation.ImageFormAlgo has been set HERE, but the incoming '
            'structure was incorrect.'.format(alg_type))
        the_sicd.ImageFormation.ImageFormAlgo = alg_type.upper()
        cond = False
    elif the_sicd.ImageFormation.ImageFormAlgo != alg_type:
        the_sicd.log_validity_warning(
            'Image formation algorithm {} populated, but ImageFormation.ImageFormAlgo populated as {}.\n'
            'ImageFormation.ImageFormAlgo has been set properly HERE, but the incoming '
            'structure was incorrect.'.format(alg_type, the_sicd.ImageFormation.ImageFormAlgo))
        the_sicd.ImageFormation.ImageFormAlgo = alg_type.upper()
        cond = False
    if the_sicd.Grid is None:
        return cond

    if the_sicd.ImageFormation.ImageFormAlgo == 'RGAZCOMP' and the_sicd.RgAzComp is not None:
        cond &= the_sicd.RgAzComp.check_parameters(
            the_sicd.Grid, the_sicd.RadarCollection, the_sicd.SCPCOA, the_sicd.Timeline, the_sicd.ImageFormation, the_sicd.GeoData)
    elif the_sicd.ImageFormation.ImageFormAlgo == 'PFA' and the_sicd.PFA is not None:
        cond &= the_sicd.PFA.check_parameters(
            the_sicd.Grid, the_sicd.SCPCOA, the_sicd.GeoData, the_sicd.Position, the_sicd.Timeline, the_sicd.RadarCollection,
            the_sicd.ImageFormation, the_sicd.CollectionInfo)
    elif the_sicd.ImageFormation.ImageFormAlgo == 'RMA':
        cond &= the_sicd.RMA.check_parameters(
            the_sicd.Grid, the_sicd.GeoData, the_sicd.RadarCollection, the_sicd.ImageFormation, the_sicd.CollectionInfo,
            the_sicd.Position)
    elif the_sicd.ImageFormation.ImageFormAlgo == 'OTHER':
        the_sicd.log_validity_warning(
            'Image formation algorithm populated as "OTHER", which inherently limits SICD analysis capability')
        cond = False
    return cond


def _validate_image_formation(the_sicd):
    """
    Validate the image formation.

    Parameters
    ----------
    the_sicd : sarpy.io.complex.sicd_elements.SICD.SICDType

    Returns
    -------
    bool
    """

    if the_sicd.ImageFormation is None:
        the_sicd.log_validity_error(
            'ImageFormation attribute is not populated, and ImageFormType is {}. This '
            'cannot be valid.'.format(the_sicd.ImageFormType))
        return False  # nothing more to be done.

    alg_types = []
    for alg in ['RgAzComp', 'PFA', 'RMA']:
        if getattr(the_sicd, alg) is not None:
            alg_types.append(alg)

    if len(alg_types) > 1:
        the_sicd.log_validity_error(
            'ImageFormation.ImageFormAlgo is set as {}, and multiple SICD image formation parameters {} are set.\n'
            'Only one image formation algorithm should be set, and ImageFormation.ImageFormAlgo '
            'should match.'.format(the_sicd.ImageFormation.ImageFormAlgo, alg_types))
        return False
    elif len(alg_types) == 0:
        if the_sicd.ImageFormation.ImageFormAlgo is None:
            the_sicd.log_validity_warning(
                'ImageFormation.ImageFormAlgo is not set, and there is no corresponding\n'
                'RgAzComp, PFA, or RMA SICD parameters set. Setting ImageFormAlgo '
                'to "OTHER".'.format(the_sicd.ImageFormation.ImageFormAlgo))
            the_sicd.ImageFormation.ImageFormAlgo = 'OTHER'
            return True
        elif the_sicd.ImageFormation.ImageFormAlgo != 'OTHER':
            the_sicd.log_validity_error(
                'No RgAzComp, PFA, or RMA SICD parameters populated, but ImageFormation.ImageFormAlgo '
                'is set as {}.'.format(the_sicd.ImageFormation.ImageFormAlgo))
            return False
        return True
    # there is exactly one algorithm type populated
    return _validate_image_form_parameters(the_sicd, alg_types[0])


def _validate_image_segment_id(the_sicd):
    """
    Validate the image segment id.

    Parameters
    ----------
    the_sicd : sarpy.io.complex.sicd_elements.SICD.SICDType

    Returns
    -------
    bool
    """

    if the_sicd.ImageFormation is None or the_sicd.RadarCollection is None:
        return False

    # get the segment identifier
    seg_id = the_sicd.ImageFormation.SegmentIdentifier
    # get the segment list
    try:
        seg_list = the_sicd.RadarCollection.Area.Plane.SegmentList
    except AttributeError:
        seg_list = None

    if seg_id is None:
        if seg_list is None:
            return True
        else:
            the_sicd.log_validity_error(
                'ImageFormation.SegmentIdentifier is not populated, but\n'
                'RadarCollection.Area.Plane.SegmentList is populated.\n'
                'ImageFormation.SegmentIdentifier should be set to identify the appropriate segment.')
            return False
    else:
        if seg_list is None:
            the_sicd.log_validity_error(
                'ImageFormation.SegmentIdentifier is populated as {},\n'
                'but RadarCollection.Area.Plane.SegmentList is not populated.'.format(seg_id))
            return False
        else:
            # let's double check that seg_id is sensibly populated
            the_ids = [entry.Identifier for entry in seg_list]
            if seg_id in the_ids:
                return True
            else:
                the_sicd.log_validity_error(
                    'ImageFormation.SegmentIdentifier is populated as {},\n'
                    'but this is not one of the possible identifiers in the\n'
                    'RadarCollection.Area.Plane.SegmentList definition {}.\n'
                    'ImageFormation.SegmentIdentifier should be set to identify the '
                    'appropriate segment.'.format(seg_id, the_ids))
                return False


def _validate_spotlight_mode(the_sicd):
    """
    Validate the spotlight mode situation.

    Parameters
    ----------
    the_sicd : sarpy.io.complex.sicd_elements.SICD.SICDType

    Returns
    -------
    bool
    """

    if the_sicd.CollectionInfo is None or the_sicd.CollectionInfo.RadarMode is None \
            or the_sicd.CollectionInfo.RadarMode.ModeType is None:
        return True

    if the_sicd.Grid is None or the_sicd.Grid.TimeCOAPoly is None:
        return True

    if the_sicd.CollectionInfo.RadarMode.ModeType == 'SPOTLIGHT' and \
            the_sicd.Grid.TimeCOAPoly.Coefs.shape != (1, 1):
        the_sicd.log_validity_error(
            'CollectionInfo.RadarMode.ModeType is SPOTLIGHT,\n'
            'but the Grid.TimeCOAPoly is not scalar - {}.\n'
            'This cannot be valid.'.format(the_sicd.Grid.TimeCOAPoly.Coefs))
        return False
    elif the_sicd.Grid.TimeCOAPoly.Coefs.shape == (1, 1) and \
            the_sicd.CollectionInfo.RadarMode.ModeType != 'SPOTLIGHT':
        the_sicd.log_validity_warning(
            'The Grid.TimeCOAPoly is scalar,\n'
            'but the CollectionInfo.RadarMode.ModeType is not SPOTLIGHT - {}.\n'
            'This is likely not valid.'.format(the_sicd.CollectionInfo.RadarMode.ModeType))
        return True
    return True


def _validate_valid_data(the_sicd):
    """
    Check that either both ValidData fields are populated, or neither.

    Parameters
    ----------
    the_sicd : sarpy.io.complex.sicd_elements.SICD.SICDType

    Returns
    -------
    bool
    """

    if the_sicd.ImageData is None or the_sicd.GeoData is None:
        return True

    cond = True
    if the_sicd.ImageData.ValidData is not None and the_sicd.GeoData.ValidData is None:
        the_sicd.log_validity_error('ValidData is populated in ImageData, but not GeoData')
        cond = False
    if the_sicd.GeoData.ValidData is not None and the_sicd.ImageData.ValidData is None:
        the_sicd.log_validity_error('ValidData is populated in GeoData, but not ImageData')
        cond = False
    return cond


def _validate_polarization(the_sicd):
    """
    Validate the polarization.

    Parameters
    ----------
    the_sicd : sarpy.io.complex.sicd_elements.SICD.SICDType

    Returns
    -------
    bool
    """

    cond = True
    pol_form = the_sicd.ImageFormation.TxRcvPolarizationProc
    if pol_form not in [entry.TxRcvPolarization for entry in the_sicd.RadarCollection.RcvChannels]:
        the_sicd.log_validity_error(
            'ImageFormation.TxRcvPolarizationProc is populated as {},\n'
            'but it not one of the tx/rcv polarizations populated for '
            'the collect'.format(pol_form))
        cond = False
    return cond


def _check_deltak(the_sicd):
    """
    Checks the deltak parameters.

    Parameters
    ----------
    the_sicd : sarpy.io.complex.sicd_elements.SICD.SICDType

    Returns
    -------
    bool
    """

    if the_sicd.Grid is None:
        return True

    x_coords, y_coords = None, None
    try:
        valid_vertices = the_sicd.ImageData.get_valid_vertex_data()
        if valid_vertices is None:
            valid_vertices = the_sicd.ImageData.get_full_vertex_data()
        x_coords = the_sicd.Grid.Row.SS * (
                    valid_vertices[:, 0] - (the_sicd.ImageData.SCPPixel.Row - the_sicd.ImageData.FirstRow))
        y_coords = the_sicd.Grid.Col.SS * (
                    valid_vertices[:, 1] - (the_sicd.ImageData.SCPPixel.Col - the_sicd.ImageData.FirstCol))
    except (AttributeError, ValueError):
        pass
    return the_sicd.Grid.check_deltak(x_coords, y_coords)


def _check_projection(the_sicd):
    """
    Checks the projection ability.

    Parameters
    ----------
    the_sicd : sarpy.io.complex.sicd_elements.SICD.SICDType
    """

    if not the_sicd.can_project_coordinates():
        the_sicd.log_validity_warning(
            'No projection can be performed for this SICD.\n'
            'In particular, no derived products can be produced.')


def _check_recommended_attributes(the_sicd):
    """
    Checks recommended attributes.

    Parameters
    ----------
    the_sicd : sarpy.io.complex.sicd_elements.SICD.SICDType
    """

    if the_sicd.Radiometric is None:
        the_sicd.log_validity_warning('No Radiometric data provided.')
    else:
        the_sicd.Radiometric.check_recommended()

    if the_sicd.Timeline is not None and the_sicd.Timeline.IPP is None:
        the_sicd.log_validity_warning(
            'No Timeline.IPP provided, so no PRF/PRI available '
            'for analysis of ambiguities.')

    if the_sicd.RadarCollection is not None and the_sicd.RadarCollection.Area is None:
        the_sicd.log_validity_info(
            'No RadarCollection.Area provided, and some tools prefer using\n'
            'a pre-determined output plane for consistent product definition.')

    if the_sicd.ImageData is not None and the_sicd.ImageData.ValidData is None:
        the_sicd.log_validity_warning(
            'No ImageData.ValidData is defined. It is recommended to populate\n'
            'this data, if validity of pixels/areas is known.')

    if the_sicd.RadarCollection is not None and the_sicd.RadarCollection.RefFreqIndex is not None:
        the_sicd.log_validity_warning(
            'A reference frequency is being used. This may affect the results of\n'
            'this validation test, because a number tests could not be performed.')


def detailed_validation_checks(the_sicd):
    """
    Assembles the suite of detailed sicd validation checks.

    Parameters
    ----------
    the_sicd : sarpy.io.complex.sicd_elements.SICD.SICDType

    Returns
    -------
    bool
    """

    out = _validate_scp_time(the_sicd)
    out &= _validate_image_formation(the_sicd)
    out &= _validate_image_segment_id(the_sicd)
    out &= _validate_spotlight_mode(the_sicd)
    out &= _validate_valid_data(the_sicd)
    out &= _validate_polarization(the_sicd)
    out &= _check_deltak(the_sicd)

    if the_sicd.SCPCOA is not None:
        out &= the_sicd.SCPCOA.check_values(the_sicd.GeoData)
    if the_sicd.Radiometric is not None:
        out &= the_sicd.Radiometric.check_parameters(the_sicd.Grid, the_sicd.SCPCOA)

    _check_projection(the_sicd)
    _check_recommended_attributes(the_sicd)
    return out
