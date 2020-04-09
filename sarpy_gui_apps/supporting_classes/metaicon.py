import logging

from tkinter import ttk
from tkinter import font
import sarpy.io.complex as sarpy_complex
from sarpy.io.complex.sicd import SICDReader
from sarpy.io.complex.sicd import SICDType
import datetime
from sarpy.geometry import geocoords
from scipy.constants import constants
import numpy as np
import math
from tkinter_gui_builder.panel_templates.image_canvas.image_canvas import ImageCanvas
import tkinter_gui_builder.utils.color_utils.color_converter as color_converter


class MetaIcon(ImageCanvas):
    __slots__ = ('fname', 'reader_object', "meta")

    def __init__(self, master):
        super().__init__(master)
        self.fname = None                              # type: str
        self.reader_object = None          # type: SICDReader
        self.meta = None           # type: SICDType
        self.layover_color = color_converter.rgb_to_hex([1, 0.65, 0])
        self.shadow_color = color_converter.rgb_to_hex([0, 0.65, 1])
        self.multipath_color = color_converter.rgb_to_hex([1, 0, 0])
        self.north_color = color_converter.rgb_to_hex([0.58, 0.82, 0.31])
        self.flight_direction_color = color_converter.rgb_to_hex([1, 1, 0])

    def create_from_fname(self, fname):
        self.fname = fname
        self.reader_object = sarpy_complex.open(fname)
        self.meta = self.reader_object.sicd_meta

        iid_line = self.get_iid_line()
        geo_line = self.get_geo_line()
        res_line = self.get_res_line()
        cdp_line = self.get_cdp_line()
        azimuth_line = self.get_azimuth_line()
        graze_line = self.get_graze_line()
        layover_line = self.get_layover_line()
        shadow_line = self.get_shadow_line()
        multipath_line = self.get_multipath_line()

        line_positions = self.get_line_positions()
        text_height = int( (line_positions[1][1] - line_positions[0][1]) * 0.7)
        canvas_font = font.Font(family='Times New Roman', size=-text_height)

        self.canvas.create_text(line_positions[0], text=iid_line, fill="white", anchor="nw", font=canvas_font)
        self.canvas.create_text(line_positions[1], text=geo_line, fill="white", anchor="nw", font=canvas_font)
        self.canvas.create_text(line_positions[2], text=res_line, fill="white", anchor="nw", font=canvas_font)
        self.canvas.create_text(line_positions[3], text=cdp_line, fill="white", anchor="nw", font=canvas_font)
        self.canvas.create_text(line_positions[4], text=azimuth_line, fill="white", anchor="nw", font=canvas_font)
        self.canvas.create_text(line_positions[5], text=graze_line, fill="white", anchor="nw", font=canvas_font)
        self.canvas.create_text(line_positions[6], text=layover_line, fill=self.layover_color, anchor="nw", font=canvas_font)
        self.canvas.create_text(line_positions[7], text=shadow_line, fill=self.shadow_color, anchor="nw", font=canvas_font)
        self.canvas.create_text(line_positions[8], text=multipath_line, fill=self.multipath_color, anchor="nw", font=canvas_font)

        # now draw the arrows
        arrow_length = self.canvas_width * 0.15
        arrows_origin = (self.canvas_width * 0.75, self.canvas_height * 0.6)

        # get arrow angles
        layover, shadow, multipath, north = self.get_arrow_layover_shadow_multipath_north_angles()

        # draw layover arrow
        self.create_new_arrow((arrows_origin[0],
                               arrows_origin[1],
                               arrows_origin[0] + arrow_length * np.cos(np.deg2rad(layover)),
                               arrows_origin[1] + arrow_length * -np.sin(np.deg2rad(layover))), fill=self.layover_color, width=2)

        self.create_new_arrow((arrows_origin[0],
                               arrows_origin[1],
                               arrows_origin[0] + arrow_length * np.cos(np.deg2rad(shadow)),
                               arrows_origin[1] + arrow_length * -np.sin(np.deg2rad(shadow))), fill=self.shadow_color, width=2)

        self.create_new_arrow((arrows_origin[0],
                               arrows_origin[1],
                               arrows_origin[0] + arrow_length * np.cos(np.deg2rad(multipath)),
                               arrows_origin[1] + arrow_length * -np.sin(np.deg2rad(multipath))), fill=self.multipath_color, width=2)

        self.create_new_arrow((arrows_origin[0],
                               arrows_origin[1],
                               arrows_origin[0] + arrow_length * np.cos(np.deg2rad(north)),
                               arrows_origin[1] + arrow_length * -np.sin(np.deg2rad(north))), fill=self.north_color, width=2)
        self.canvas.create_text((arrows_origin[0] + arrow_length*1.3 * np.cos(np.deg2rad(north)),
                                 arrows_origin[1] + arrow_length*1.3 * -np.sin(np.deg2rad(north))),
                                text="N",
                                fill=self.north_color,
                                font=canvas_font)

        flight_direction_arrow_start = (self.canvas_width*0.65, self.canvas_height*0.9)
        flight_direction_arrow_end = (self.canvas_width*0.95, flight_direction_arrow_start[1])
        if self.meta.SCPCOA.SideOfTrack == "R":
            self.create_new_arrow((flight_direction_arrow_start[0],
                                   flight_direction_arrow_start[1],
                                   flight_direction_arrow_end[0],
                                   flight_direction_arrow_end[1]), fill=self.flight_direction_color, width=3)
        else:
            self.create_new_arrow((flight_direction_arrow_end[0],
                                   flight_direction_arrow_end[1],
                                   flight_direction_arrow_start[0],
                                   flight_direction_arrow_start[1]), fill=self.flight_direction_color, width=3)
        self.canvas.create_text((flight_direction_arrow_start[0] - self.canvas_width * 0.04,
                                 flight_direction_arrow_start[1]),
                                text = "R",
                                fill = self.flight_direction_color,
                                font=canvas_font)

    def get_line_positions(self, margin_percent=5):
        n_lines = 9
        height = self.canvas_height
        width = self.canvas_width
        margin = height * (margin_percent*0.01*2)
        top_margin = margin/2
        height_w_margin = height - margin
        y_positions = np.linspace(0, height_w_margin, n_lines+1)
        y_positions = y_positions + top_margin
        y_positions = y_positions[0:-1]
        x_positions = width * margin_percent * 0.01

        xy_positions = []
        for pos in y_positions:
            xy_positions.append((x_positions, pos))
        return xy_positions

    def get_multipath_line(self, n_decimals=1):
        multipath = self._get_multipath()
        return "Multipath: " + str(round(multipath, n_decimals)) + " deg"

    def _get_multipath(self):
        if hasattr(self.meta, "SCPCOA"):
            multipath_ground = self._get_multipath_ground()
            multipath = np.mod(self.meta.SCPCOA.AzimAng - 180 + multipath_ground, 360)
            return multipath

    def _get_multipath_ground(self):
        if hasattr(self.meta, "SCPCOA"):
            multipath_ground = np.rad2deg(-1*math.atan(math.tan(np.deg2rad(self.meta.SCPCOA.TwistAng)) *
                                                       math.sin(np.deg2rad(self.meta.SCPCOA.GrazeAng))))
            return multipath_ground

    def get_azimuth_line(self, n_decimals=1):
        azimuth = self._get_azimuth()
        return "Azimuth: " + str(round(self.meta.SCPCOA.AzimAng, n_decimals)) + " deg"

    def _get_azimuth(self):
        if hasattr(self.meta, "SCPCOA"):
            return self.meta.SCPCOA.AzimAng

    def get_graze_line(self, n_decimals=1):
        if hasattr(self.meta, "SCPCOA"):
            return "Graze: " + str(round(self.meta.SCPCOA.GrazeAng, n_decimals)) + " deg"

    def get_layover_line(self, n_decimals=1):
        layover = self._get_layover()
        return "Layover: " + str(round(layover, n_decimals)) + " deg"

    def _get_layover(self):
        if hasattr(self.meta, "SCPCOA"):
            return self.meta.SCPCOA.LayoverAng

    def get_shadow_line(self, n_decimals=1):
        shadow = self._get_shadow()
        return "Shadow: " + str(round(shadow, n_decimals)) + " deg"

    def _get_shadow(self):
        if hasattr(self.meta, "SCPCOA"):
            azimuth = self.meta.SCPCOA.AzimAng
            shadow = np.mod(azimuth - 180, 360)
            return shadow

    def get_cdp_line(self):
        collect_start, collect_duration = self.get_timings()
        cdp_line = "CDP: " + "{:.1f}".format(collect_duration) + " s"
        polarization = self.get_polarization()
        if polarization:
            cdp_line = cdp_line + " / POL: " + polarization[0] + polarization[2]
        return cdp_line

    def get_res_line(self):
        res_line = "IPR: "
        if hasattr(self.meta, "Grid"):
            az_ipr = self.meta.Grid.Col.ImpRespWid / constants.foot
            rg_ipr = self.meta.Grid.Row.ImpRespWid / constants.foot
            if az_ipr/rg_ipr - 1 < 0.2:
                ipr = (az_ipr + rg_ipr)/2.0
                ipr_str = "{:.1f}".format(ipr)
                res_line = res_line + ipr_str + " ft"
            else:
                ipr_str = "{:.1f}".format(az_ipr) + "/" + "{:.1f}".format(rg_ipr)
                res_line = res_line + ipr_str + "ft(A/R)"
        else:
            try:
                bw = self.meta.RadarCollection.Waveform.WFParameters.TxRFBandwidth / 1e6
                res_line = res_line + "{:.0f}".format(bw) + " MHz"
            except Exception as e:
                logging.error("no bandwidth field {}".format(e))
        if self.get_rniirs():
            res_line = res_line + " RNIIRS: " + str(self.get_rniirs())
        return res_line

    def get_iid_line(self):
        if hasattr(self.meta, "Timeline"):
            try:
                date_str = self.meta.Timeline.CollectStart.astype(datetime.datetime).strftime("%d%b%y").upper()
                collector_name_str = self.meta.CollectionInfo.CollectorName
                if len(collector_name_str) > 4:
                    collector_name_str = collector_name_str[0:4]
                iid_line = date_str + " " + collector_name_str
            except Exception as e:
                logging.error("date and/or collector name are not populated {}".format(e))

        elif hasattr(self.meta, "Global"):
            try:
                iid_line = self.meta.CollectionInfo.CoreName
                if len(iid_line) > 16:
                    iid_line = iid_line[0:16]
            except Exception as e:
                logging.error("no field in meta.CollectionInfo.CoreName {}".format(e))
        else:
            iid_line = ""
        if hasattr(self.meta.Timeline, "CollectStart"):
            iid_datestr = self.meta.Timeline.CollectStart.astype(datetime.datetime).strftime("%H%MZ")
            iid_line = iid_line + " / " + iid_datestr
        return iid_line

    def get_geo_line(self):
        lat, lon = self.get_geo_lon_lat()
        geo_line = "Geo: " + "{:.4f}".format(lat) + "/" + "{:.4f}".format(lon)
        return geo_line

    def get_timings(self):
        if hasattr(self.meta, "Timeline"):
            if hasattr(self.meta.Timeline, "CollectStart"):
                collect_start = self.meta.Timeline.CollectStart
            if hasattr(self.meta.Timeline, "CollectDuration"):
                collect_duration = self.meta.Timeline.CollectDuration
        elif hasattr(self.meta, "Global"):
            try:
                collect_start = self.meta.Global.CollectStart
            except Exception as e:
                logging.error("No field found in Global.CollectStart.  {}".format(e))
        # TODO: implement collect duration for the case where vbmeta.TxTime is used.
        return collect_start, collect_duration

    # TODO
    def get_country_code(self):
        pass

    def get_polarization(self):
        pol = None
        if hasattr(self.meta, "ImageFormation"):
            try:
                pol = self.meta.ImageFormation.TxRcvPolarizationProc
            except Exception as e:
                logging.error("polarization not found in meta.ImageFormation.TxRcvPolarizationProc {}".format(e))
        elif hasattr(self.meta, "RadarCollection"):
            try:
                pol = self.meta.RadarCollection.RcvChannels.ChanParameters.TxRcvPolarization
            except Exception as e:
                logging.error("No field found in meta.RadarCollection.RcvChannels.ChanParameters.TxRcvPolarization {}".format(e))
        return pol

    def get_geo_lon_lat(self):
        if hasattr(self.meta, "GeoData"):
            try:
                scp = [self.meta.GeoData.SCP.ECF.X, self.meta.GeoData.SCP.ECF.Y, self.meta.GeoData.SCP.ECF.Z]
            except Exception as e:
                logging.error("Unabel to get geolocation data in ECF form {}".format(e))
        elif hasattr(self.meta, "SRP"):
            if self.meta.SRP.SRPType == "FIXEDPT":
                scp = self.meta.SRP.FIXEDPT.SRPPT
                scp = [scp.X, scp.Y, scp.Z]
            elif self.meta.SRP.SRPType == "PVTPOLY":
                # TODO: implement this for the case where we need to do a polynomial
                pass
        try:
            lla = geocoords.ecf_to_geodetic(scp)
            lat = lla[0]
            lon = lla[1]
        except Exception as e:
            logging.error("could not find latitude and longitude information in the SICD metadata. {}".format(e))
        return lat, lon
        # TODO: implement vbmeta version
        # TODO: implement a version of latlonstr from the MATLAB repo in sarpy Geometry, if this is important.

    def get_rniirs(self):
        if hasattr(self.meta, 'CollectionInfo') and hasattr(self.meta.CollectionInfo, 'Parameter'):
            return self.meta.CollectionInfo.Parameters['RNIIRS']
        else:
            return None

    def get_arrow_layover_shadow_multipath_north_angles(self):
        # TODO: check additional parameter for GroundProject and ensure it's false
        azimuth = self._get_azimuth()
        if hasattr(self.meta, "Grid") or self.meta.Grid.ImagePlane == 'SLANT':
            shadow = azimuth - 180 - self._get_multipath_ground()
            multipath = azimuth - 180
            layover = self._get_layover() - self._get_multipath_ground()

        layover = 90 - (layover - azimuth)
        shadow = 90 - (shadow - azimuth)
        north = azimuth + 90
        multipath = north - multipath
        return layover, shadow, multipath, north