"""
This module provides tools for creating kmz products for a CPHD type element.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Valkyrie Systems Corporation"

import logging
import os

import numpy as np

import sarpy.visualization.kmz_product_creation as kpc
from sarpy.io.kml import Document
from sarpy.visualization import kmz_utils

logger = logging.getLogger(__name__)


def _create_cphd_styles(kmz_document):
    kmz_utils.add_antenna_styles(kmz_document)

    # iarp - intended for basic point clamped to ground
    label = {"color": "ff50c0c0", "scale": "1.0"}
    icon = {
        "color": "ff5050c0",
        "scale": "1.5",
        "icon_ref": "http://maps.google.com/mapfiles/kml/shapes/shaded_dot.png",
    }
    kmz_document.add_style("iarp_high", label_style=label, icon_style=icon)
    label["scale"] = "0.75"
    icon["scale"] = "1.0"
    kmz_document.add_style("iarp_low", label_style=label, icon_style=icon)
    kmz_document.add_style_map("iarp", "iarp_high", "iarp_low")

    # srp position style - intended for gx track
    line = {"color": "ff50ff50", "width": "1.5"}
    label = {"color": "ffc0c0c0", "scale": "0.5"}
    icon = {
        "scale": "2.0",
        "icon_ref": "http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png",
    }
    poly = {"color": "a050ff50"}
    kmz_document.add_style(
        "srp_high", line_style=line, label_style=label, icon_style=icon, poly_style=poly
    )
    line["width"] = "1.0"
    icon["scale"] = "1.0"
    poly = {"color": "7050ff50"}
    kmz_document.add_style(
        "srp_low", line_style=line, label_style=label, icon_style=icon, poly_style=poly
    )
    kmz_document.add_style_map("srp", "srp_high", "srp_low")

    kmz_utils.add_polygon_style(
        kmz_document,
        "globalimagearea",
        bbggrr="aa00ff",
        low_aa="00",
        low_width="1.0",
        high_aa="00",
        high_width="1.5",
    )

    kmz_utils.add_polygon_style(
        kmz_document,
        "channelimagearea",
        bbggrr="00aa7f",
        low_aa="70",
        low_width="1.0",
        high_aa="a0",
        high_width="1.5",
    )


def imagearea_kml_coord(scenecoords_node, imagearea_node):
    """Create KML coords of an imagearea footprint"""
    if imagearea_node.Polygon is not None:
        verts = [
            vert.get_array()
            for vert in sorted(imagearea_node.Polygon, key=lambda x: x.index)
        ]
    else:
        x1, y1 = imagearea_node.X1Y1.get_array()
        x2, y2 = imagearea_node.X2Y2.get_array()
        verts = [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]
    verts.append(verts[0])
    return _xy_to_kml_coord(scenecoords_node, verts)


def cphd_create_kmz_view(reader, output_directory, file_stem="view"):
    """
    Create a kmz view for the reader contents.

    Parameters
    ----------
    reader : CPHDTypeReader
    output_directory : str
    file_stem : str

    Returns
    -------
    None

    Examples
    --------
    .. code-block:: python

        import logging
        logger = logging.getLogger('sarpy')
        logger.setLevel('INFO')

        import os
        import sarpy.io.phase_history
        from sarpy.visualization.cphd_kmz_product_creation import cphd_create_kmz_view

        test_root = '<root directory>'
        reader = sarpy.io.phase_history.open(os.path.join(test_root, '<file name>>'))
        cphd_create_kmz_view(reader, test_root,
                             file_stem='View-<something descriptive>')

    """

    def add_global(kmz_doc, root):
        logger.info("Adding global to kmz.")
        folder = kmz_doc.add_container(
            par=root, the_type="Folder", name="Global", description="Global Metadata"
        )
        iarp_llh = reader.cphd_meta.SceneCoordinates.IARP.LLH.get_array()
        frm = "{1:0.8f},{0:0.8f},0"  # SICD version throws away height
        coords = frm.format(*iarp_llh)
        placemark = kmz_doc.add_container(
            par=folder, name="IARP", description="IARP", styleUrl="#iarp"
        )
        kmz_doc.add_point(coords, par=placemark, altitudeMode="absolute")

        ia_coords = imagearea_kml_coord(
            reader.cphd_meta.SceneCoordinates,
            reader.cphd_meta.SceneCoordinates.ImageArea,
        )
        placemark = kmz_doc.add_container(
            par=folder,
            name="ImageArea",
            description="ImageArea",
            styleUrl="#globalimagearea",
        )
        kmz_doc.add_polygon(
            " ".join(ia_coords),
            par=placemark,
            name="ImageArea",
            altitudeMode="absolute",
        )

    def add_channel(kmz_doc, root, channel_name):
        channel_names = [chan.Identifier for chan in reader.cphd_meta.Data.Channels]
        channel_index = channel_names.index(channel_name)
        pvp_array = reader.read_pvp_array(channel_index)
        logger.info(f"Adding channel '{channel_name}' to kmz.")

        if "SIGNAL" in pvp_array.dtype.names:
            signal = pvp_array["SIGNAL"]
        else:
            num_vectors = reader.cphd_meta.Data.Channels[channel_index].NumVectors
            signal = np.ones(num_vectors)

        num_subselect = 24
        indices = np.where(signal == 1)[0]
        geom_indices = indices[
            np.round(
                np.linspace(0, indices.size - 1, num_subselect, endpoint=True)
            ).astype(int)
        ]

        times = (
            pvp_array["TxTime"][geom_indices] + pvp_array["RcvTime"][geom_indices]
        ) / 2.0
        arp_pos = (
            pvp_array["TxPos"][geom_indices] + pvp_array["RcvPos"][geom_indices]
        ) / 2.0
        srp_pos = pvp_array["SRPPos"][geom_indices]
        collection_start = reader.cphd_meta.Global.Timeline.CollectionStart.astype(
            "datetime64[us]"
        )
        whens = collection_start + (times * 1e6).astype("timedelta64[us]")
        whens = [str(time) + "Z" for time in whens]

        time_args = {"beginTime": whens[0], "endTime": whens[-1]}

        folder = kmz_doc.add_container(
            par=root,
            the_type="Folder",
            name=f"Channel {channel_name}",
            description=f"Channel {channel_name}",
            when=whens[0],
        )
        arp_coords = kmz_utils.ecef_to_kml_coord(arp_pos)
        placemark = kmz_doc.add_container(
            par=folder,
            name=channel_name,
            description=f"aperture position for channel {channel_name}",
            styleUrl="#arp",
            **time_args,
        )
        kmz_doc.add_gx_track(
            arp_coords,
            whens,
            par=placemark,
            extrude=True,
            tesselate=True,
            altitudeMode="absolute",
        )

        srp_coords = kmz_utils.ecef_to_kml_coord(srp_pos)
        placemark = kmz_doc.add_container(
            par=folder,
            name="SRP",
            description=f"stabilization reference point for channel {channel_name}",
            styleUrl="#srp",
            **time_args,
        )
        kmz_doc.add_gx_track(
            srp_coords,
            whens,
            par=placemark,
            extrude=True,
            tesselate=True,
            altitudeMode="absolute",
        )

        if reader.cphd_meta.Channel.Parameters[channel_index].ImageArea is not None:
            ia_coords = imagearea_kml_coord(
                reader.cphd_meta.SceneCoordinates,
                reader.cphd_meta.Channel.Parameters[channel_index].ImageArea,
            )
            placemark = kmz_doc.add_container(
                par=folder,
                name="ImageArea",
                description=f"ImageArea for channel {channel_name}",
                styleUrl="#channelimagearea",
            )
            kmz_doc.add_polygon(
                " ".join(ia_coords),
                par=placemark,
                name=f"ImageArea for channel {channel_name}",
                altitudeMode="absolute",
            )

        chan_params = reader.cphd_meta.Channel.Parameters[channel_index]
        if chan_params.Antenna is None:
            return

        antenna_folder = kmz_doc.add_container(
            the_type="Folder",
            par=folder,
            name="Antenna",
            description=f"Antenna Aiming for channel {channel_name}",
        )
        boresight_folder = kmz_doc.add_container(
            the_type="Folder",
            par=antenna_folder,
            name="Boresights",
            description=f"Boresights for channel {channel_name}",
        )
        footprint_folder = kmz_doc.add_container(
            the_type="Folder",
            par=antenna_folder,
            name="-3dB Footprints",
            description=f"Beam Footprints for channel {channel_name}",
        )

        footprint_labels = {
            "start": indices[0],
            "middle": indices[len(indices) // 2],
            "end": indices[-1],
        }

        for txrcv in ("Tx", "Rcv"):
            aiming = antenna_aiming(
                reader.cphd_meta.Antenna,
                pvp_array,
                txrcv=txrcv,
                apc_id=getattr(chan_params.Antenna, f"{txrcv}APCId"),
                antpat_id=getattr(chan_params.Antenna, f"{txrcv}APATId"),
            )

            for boresight_type in ("mechanical", "electrical"):
                visibility = txrcv == "Rcv"  # only display Rcv by default
                name = f"{txrcv} {boresight_type} boresight"

                on_earth_ecf = np.asarray(
                    [
                        kmz_utils.ray_intersect_earth(apc_pos, along)
                        for apc_pos, along in zip(
                            aiming["raw"]["positions"][geom_indices],
                            aiming[boresight_type][geom_indices],
                        )
                    ]
                )

                placemark = kmz_doc.add_container(
                    par=boresight_folder,
                    name=name,
                    description=f"{name} for channel {channel_name}<br><br>Highlighted edge indicates start time",
                    styleUrl=f"#{boresight_type}_boresight",
                    visibility=visibility,
                )
                boresight_coords = kmz_utils.ecef_to_kml_coord(on_earth_ecf)
                kmz_utils.add_los_polygon(
                    kmz_doc, placemark, arp_coords, boresight_coords
                )

            array_gain_poly = aiming["raw"]["pattern"].Array.GainPoly
            element_gain_poly = aiming["raw"]["pattern"].Element.GainPoly
            footprints = kmz_utils.make_beam_footprints(
                aiming,
                footprint_labels,
                array_gain_poly,
                element_gain_poly,
                contour_level=-3,
            )
            for when, this_footprint in footprints.items():
                name = f"{txrcv} beam footprint @ {when}"
                timestamp = (
                    str(
                        collection_start
                        + (this_footprint["time"] * 1e6).astype("timedelta64[us]")
                    )
                    + "Z"
                )
                placemark = kmz_doc.add_container(
                    par=footprint_folder,
                    name=name,
                    description=f"{name} for channel {channel_name}",
                    styleUrl=f"#{txrcv.lower()}_beam_footprint",
                    visibility=True,
                    when=timestamp,
                )
                coords = kmz_utils.ecef_to_kml_coord(this_footprint["contour"])
                kmz_doc.add_polygon(
                    " ".join(coords),
                    par=placemark,
                )

    kmz_file = os.path.join(output_directory, f"{file_stem}_cphd.kmz")
    with prepare_kmz_file(kmz_file, name=reader.file_name) as kmz_doc:
        root = kmz_doc.add_container(
            the_type="Folder", name=reader.cphd_meta.CollectionID.CoreName
        )
        add_global(kmz_doc, root)
        for chan in reader.cphd_meta.Data.Channels:
            add_channel(kmz_doc, root, channel_name=chan.Identifier)


def prepare_kmz_file(file_name, **args):
    """
    Prepare a kmz document and archive for exporting.

    Parameters
    ----------
    file_name : str
    args
        Passed through to the Document constructor.

    Returns
    -------
    Document
    """

    document = Document(file_name=file_name, **args)
    kpc._create_sicd_styles(document)
    _create_cphd_styles(document)
    return document


def _xy_to_kml_coord(scenecoords_node, xy):
    """Convert a ReferenceSurface XY location to a kml coordinate"""
    xy = np.atleast_2d(xy)
    if scenecoords_node.ReferenceSurface.Planar is not None:
        iarp_ecf = scenecoords_node.IARP.ECF.get_array()
        uiax = scenecoords_node.ReferenceSurface.Planar.uIAX.get_array()
        uiay = scenecoords_node.ReferenceSurface.Planar.uIAY.get_array()
        xy_ecf = iarp_ecf + xy[:, 0, np.newaxis] * uiax + xy[:, 1, np.newaxis] * uiay
    else:
        raise NotImplementedError
    return kmz_utils.ecef_to_kml_coord(xy_ecf)


def antenna_aiming(antenna_node, pvp_array, *, txrcv, apc_id, antpat_id):
    """Compile antenna aiming metadata"""
    if antenna_node is None:
        return {}

    apcs = {apc.Identifier: apc for apc in antenna_node.AntPhaseCenter}
    acfs = {acf.Identifier: acf for acf in antenna_node.AntCoordFrame}
    patterns = {antpat.Identifier: antpat for antpat in antenna_node.AntPattern}

    positions = pvp_array[f"{txrcv}Pos"]
    times = pvp_array[f"{txrcv}Time"]
    if {f"{txrcv}AC{d}" for d in "XY"}.issubset(pvp_array.dtype.names):
        uacx = pvp_array[f"{txrcv}ACX"]
        uacy = pvp_array[f"{txrcv}ACY"]
    else:
        acf_id = apcs[apc_id].ACFId
        uacx = acfs[acf_id].XAxisPoly(times)
        uacy = acfs[acf_id].YAxisPoly(times)
    uacz = np.cross(uacx, uacy)

    pointing = {
        "antpat_id": antpat_id,
        "raw": {
            "positions": positions,
            "times": times,
            "uacx": uacx,
            "uacy": uacy,
            "uacz": uacz,
            "pattern": patterns[antpat_id],
        },
        "mechanical": uacz,
    }

    if f"{txrcv}EB" in pvp_array.dtype.names:
        ebpvp = pvp_array[f"{txrcv}EB"]
        eb_dcx = ebpvp[:, 0]
        eb_dcy = ebpvp[:, 1]
    else:
        eb_dcx = patterns[antpat_id].EB.DCXPoly(times)
        eb_dcy = patterns[antpat_id].EB.DCYPoly(times)

    pointing["raw"]["eb_dcx"] = eb_dcx
    pointing["raw"]["eb_dcy"] = eb_dcy

    pointing["electrical"] = kmz_utils.acf_to_ecef(eb_dcx, eb_dcy, uacx, uacy)

    return pointing
