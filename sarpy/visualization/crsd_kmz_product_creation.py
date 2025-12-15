"""
This module provides tools for creating kmz products for a CRSD type element.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Valkyrie Systems Corporation"

import logging
import numpy as np
import os
from deprecated import deprecated

import sarpy.visualization.cphd_kmz_product_creation as cphd_kpc
from sarpy.visualization import kmz_utils

logger = logging.getLogger(__name__)

@deprecated("sarpy's CRSD implementation is deprecated. Please use SARKit.")
def crsd_create_kmz_view(reader, output_directory, file_stem="view"):
    """
    Create a kmz view for the reader contents.

    Parameters
    ----------
    reader : CRSDTypeReader
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
        import sarpy.io.received.converter
        from sarpy.visualization.crsd_kmz_product_creation import crsd_create_kmz_view

        test_root = '<root directory>'
        reader = sarpy.io.received.converter.open_received(os.path.join(test_root, '<file name>>'))
        crsd_create_kmz_view(reader, test_root,
                             file_stem='View-<something descriptive>')

    """

    def add_global(kmz_doc, root):
        logger.info("Adding global to kmz.")
        folder = kmz_doc.add_container(
            par=root, the_type="Folder", name="Global", description="Global Metadata"
        )
        crp_llh = reader.crsd_meta.ReferenceGeometry.CRP.LLH.get_array()
        frm = "{1:0.8f},{0:0.8f},0"  # SICD version throws away height
        coords = frm.format(*crp_llh)
        placemark = kmz_doc.add_container(
            par=folder, name="CRP", description="CRP", styleUrl="#iarp"
        )
        kmz_doc.add_point(coords, par=placemark, altitudeMode="absolute")

        if reader.crsd_meta.SceneCoordinates is not None:
            ia_coords = cphd_kpc.imagearea_kml_coord(
                reader.crsd_meta.SceneCoordinates,
                reader.crsd_meta.SceneCoordinates.ImageArea,
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
        channel_names = [chan.Identifier for chan in reader.crsd_meta.Data.Channels]
        channel_index = channel_names.index(channel_name)
        pvp_array = reader.read_pvp_array(channel_index)
        logger.info(f"Adding channel '{channel_name}' to kmz.")
        channel_folder = kmz_doc.add_container(
            par=root,
            the_type="Folder",
            name=f"Channel {channel_name}",
            description=f"Channel {channel_name}",
        )

        chan_params = reader.crsd_meta.Channel.Parameters[channel_index]
        sar_imaging_node = chan_params.SARImaging

        for txrcv, platform_label in [("Tx", "Illuminator"), ("Rcv", "Collector")]:
            if txrcv == "Tx" and sar_imaging_node is None:
                continue

            pos_pvp = pvp_array[f"{txrcv}Pos"]
            time_pvp = pvp_array[f"{txrcv}Time"]
            num_subselect = 24
            indices = np.where(
                np.logical_and(np.isfinite(time_pvp), np.isfinite(pos_pvp).all(axis=1))
            )[0]
            indices = indices[
                np.round(
                    np.linspace(0, indices.size - 1, num_subselect, endpoint=True)
                ).astype(int)
            ]
            time_pvp = time_pvp[indices]
            apc_pos = pos_pvp[indices]
            collection_start = (
                reader.crsd_meta.Global.Timeline.CollectionRefTime.astype(
                    "datetime64[us]"
                )
            )
            whens = collection_start + (time_pvp * 1e6).astype("timedelta64[us]")
            whens = [str(time) + "Z" for time in whens]

            time_args = {"beginTime": whens[0], "endTime": whens[-1]}

            platform_folder = kmz_doc.add_container(
                the_type="Folder",
                par=channel_folder,
                name=platform_label,
                description=f"channel {channel_name}: {platform_label}",
                when=whens[0],
            )

            # Add SarImaging/ImageArea for Tx only if available
            if (
                txrcv == "Tx"
                and sar_imaging_node.ImageArea is not None
                and reader.crsd_meta.SceneCoordinates is not None
            ):
                ia_coords = cphd_kpc.imagearea_kml_coord(
                    reader.crsd_meta.SceneCoordinates, sar_imaging_node.ImageArea
                )
                placemark = kmz_doc.add_container(
                    par=platform_folder,
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

            # Add APC
            apc_coords = kmz_utils.ecef_to_kml_coord(apc_pos)
            placemark = kmz_doc.add_container(
                par=platform_folder,
                name=f"{channel_name} > {txrcv}",
                description=platform_label,
                styleUrl="#arp",
                **time_args,
            )
            kmz_doc.add_gx_track(
                apc_coords,
                whens,
                par=placemark,
                extrude=True,
                tesselate=True,
                altitudeMode="absolute",
            )

            if txrcv == "Tx":
                if (
                    chan_params.SARImaging is not None
                    and chan_params.SARImaging.TxAntenna is not None
                ):
                    apc_id = chan_params.SARImaging.TxAntenna.TxAPCId
                    antpat_id = chan_params.SARImaging.TxAntenna.TxAPATId
                else:
                    return
            else:
                if chan_params.RcvAntenna is not None:
                    apc_id = chan_params.RcvAntenna.RcvAPCId
                    antpat_id = chan_params.RcvAntenna.RcvAPATId
                else:
                    return

            antenna_folder = kmz_doc.add_container(
                the_type="Folder",
                par=platform_folder,
                name="Antenna",
                description=f"Antenna Aiming for channel {channel_name}",
            )
            boresight_folder = kmz_doc.add_container(
                the_type="Folder",
                par=antenna_folder,
                name="Boresights",
                description=f"Boresights for channel {channel_name}",
            )

            aiming = cphd_kpc.antenna_aiming(
                reader.crsd_meta.Antenna,
                pvp_array,
                txrcv=txrcv,
                apc_id=apc_id,
                antpat_id=antpat_id,
            )

            for boresight_type in ("mechanical", "electrical"):
                visibility = txrcv == "Rcv"  # only display Rcv by default
                name = f"{txrcv} {boresight_type} boresight"

                on_earth_ecf = np.asarray(
                    [
                        kmz_utils.ray_intersect_earth(apc_pos, along)
                        for apc_pos, along in zip(
                            aiming["raw"]["positions"][indices],
                            aiming[boresight_type][indices],
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
                kmz_utils.add_los_polygon(kmz_doc, placemark, apc_coords, boresight_coords)

    kmz_file = os.path.join(output_directory, f"{file_stem}_crsd.kmz")
    with cphd_kpc.prepare_kmz_file(kmz_file, name=reader.file_name) as kmz_doc:
        root = kmz_doc.add_container(
            the_type="Folder", name=reader.crsd_meta.CollectionID.CoreName
        )
        add_global(kmz_doc, root)
        for chan in reader.crsd_meta.Data.Channels:
            add_channel(kmz_doc, root, channel_name=chan.Identifier)
