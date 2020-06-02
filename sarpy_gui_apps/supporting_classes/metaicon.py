import logging

from tkinter import font
import datetime
import sarpy.io.complex as sarpy_complex
from sarpy.io.complex.sicd import SICDType
from sarpy.geometry import geocoords
from scipy.constants import constants
import numpy
from tkinter_gui_builder.panel_templates.image_canvas_panel.image_canvas_panel import ImageCanvasPanel
import tkinter_gui_builder.utils.color_utils.color_converter as color_converter
from sarpy.geometry import latlon
from tkinter_gui_builder.image_readers.numpy_image_reader import NumpyImageReader


class MetaIcon(ImageCanvasPanel):
    __slots__ = ('fname', 'reader_object', "meta")

    def __init__(self, master):
        super().__init__(master)
        self.parent = master
        self.fname = None                              # type: str
        self.meta = None           # type: SICDType
        self.layover_color = color_converter.rgb_to_hex([1, 0.65, 0])
        self.shadow_color = color_converter.rgb_to_hex([0, 0.65, 1])
        self.multipath_color = color_converter.rgb_to_hex([1, 0, 0])
        self.north_color = color_converter.rgb_to_hex([0.58, 0.82, 0.31])
        self.flight_direction_color = color_converter.rgb_to_hex([1, 1, 0])

        self.parent.protocol("WM_DELETE_WINDOW", self.close_window)

    def close_window(self):
        self.parent.withdraw()

    def create_from_sicd(self,
                         sicd_meta,     # type: SICDType
                         ):
        metaicon_background = numpy.zeros((self.canvas.canvas_height, self.canvas.canvas_width))
        numpy_reader = NumpyImageReader(metaicon_background)
        self.canvas.set_image_reader(numpy_reader)

        self.meta = sicd_meta
        iid_line = self.get_iid_line()
        geo_line = self.get_geo_line()
        res_line = self.get_res_line()
        cdp_line = self.get_cdp_line()
        azimuth_line = self._create_angle_line_text("azimuth", n_decimals=1)
        graze_line = self._create_angle_line_text("graze", n_decimals=1)
        layover_line = self._create_angle_line_text("layover", n_decimals=0)
        shadow_line = self._create_angle_line_text("shadow", n_decimals=0)
        multipath_line = self._create_angle_line_text("multipath", n_decimals=0)

        line_positions = self._get_line_positions()
        text_height = int((line_positions[1][1] - line_positions[0][1]) * 0.8)
        canvas_font = font.Font(family='Times New Roman', size=-text_height)

        self.canvas.create_text(line_positions[0], text=iid_line, fill="white", anchor="nw", font=canvas_font)
        self.canvas.create_text(line_positions[1], text=geo_line, fill="white", anchor="nw", font=canvas_font)
        self.canvas.create_text(line_positions[2], text=res_line, fill="white", anchor="nw", font=canvas_font)
        self.canvas.create_text(line_positions[3], text=cdp_line, fill="white", anchor="nw", font=canvas_font)
        self.canvas.create_text(line_positions[4], text=azimuth_line, fill="white", anchor="nw", font=canvas_font)
        self.canvas.create_text(line_positions[5], text=graze_line, fill="white", anchor="nw", font=canvas_font)
        self.canvas.create_text(line_positions[6], text=layover_line, fill=self.layover_color, anchor="nw",
                                font=canvas_font)
        self.canvas.create_text(line_positions[7], text=shadow_line, fill=self.shadow_color, anchor="nw",
                                font=canvas_font)
        self.canvas.create_text(line_positions[8], text=multipath_line, fill=self.multipath_color, anchor="nw",
                                font=canvas_font)

        # draw layover arrow
        self.draw_arrow("layover")
        self.draw_arrow("shadow")
        self.draw_arrow("multipath")
        self.draw_arrow("north")

        flight_direction_arrow_start = (self.canvas.canvas_width * 0.65, self.canvas.canvas_height * 0.9)
        flight_direction_arrow_end = (self.canvas.canvas_width * 0.95, flight_direction_arrow_start[1])
        if self.meta.SCPCOA.SideOfTrack == "R":
            self.canvas.create_new_arrow((flight_direction_arrow_start[0],
                                   flight_direction_arrow_start[1],
                                   flight_direction_arrow_end[0],
                                   flight_direction_arrow_end[1]), fill=self.flight_direction_color, width=3)
        else:
            self.canvas.create_new_arrow((flight_direction_arrow_end[0],
                                   flight_direction_arrow_end[1],
                                   flight_direction_arrow_start[0],
                                   flight_direction_arrow_start[1]), fill=self.flight_direction_color, width=3)
        self.canvas.create_text((flight_direction_arrow_start[0] - self.canvas.canvas_width * 0.04,
                                 flight_direction_arrow_start[1]),
                                text="R",
                                fill=self.flight_direction_color,
                                font=canvas_font)

    def create_from_fname(self, fname):
        self.fname = fname
        reader_object = sarpy_complex.open(fname)
        self.create_from_sicd(reader_object.sicd_meta)

    def _get_line_positions(self, margin_percent=5):
        n_lines = 9
        height = self.canvas.canvas_height
        width = self.canvas.canvas_width
        margin = height * (margin_percent*0.01*2)
        top_margin = margin/2
        height_w_margin = height - margin
        y_positions = numpy.linspace(0, height_w_margin, n_lines+1)
        y_positions = y_positions + top_margin
        y_positions = y_positions[0:-1]
        x_positions = width * margin_percent * 0.01

        xy_positions = []
        for pos in y_positions:
            xy_positions.append((x_positions, pos))
        return xy_positions

    def _create_angle_line_text(self,
                                angle_type,  # type: str
                                n_decimals,  # type: int
                                ):
        if angle_type.lower() == "layover":
            angle = self.meta.SCPCOA.LayoverAng
        elif angle_type.lower() == "shadow":
            angle = self.meta.SCPCOA.Shadow
        elif angle_type.lower() == "multipath":
            angle = self.meta.SCPCOA.Multipath
        elif angle_type.lower() == "azimuth":
            angle = self.meta.SCPCOA.AzimAng
        elif angle_type.lower() == "graze":
            angle = self.meta.SCPCOA.GrazeAng

        angle_description_text = angle_type.lower().capitalize()

        if n_decimals > 0:
            return angle_description_text + ": " + str(round(angle, n_decimals)) + "\xB0"
        else:
            return angle_description_text + ": " + str(int(round(angle))) + "\xB0"

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
        iid_line = ""
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
            pass
        if hasattr(self.meta.Timeline, "CollectStart"):
            iid_datestr = self.meta.Timeline.CollectStart.astype(datetime.datetime).strftime("%H%MZ")
            iid_line = iid_line + " / " + iid_datestr
        return iid_line

    def get_geo_line(self):
        lat, lon = self.get_geo_lon_lat()
        lat_str = latlon.string(lat, "lat", include_symbols=False)
        lon_str = latlon.string(lon, "lon", include_symbols=False)
        geo_line = "Geo: " + lat_str + "/" + lon_str
        return geo_line

    def get_timings(self):
        if hasattr(self.meta, "Timeline"):
            if hasattr(self.meta.Timeline, "CollectStart"):
                collect_start = self.meta.Timeline.CollectStart
            if hasattr(self.meta.Timeline, "CollectDuration"):
                collect_duration = self.meta.Timeline.CollectDuration
            return collect_start, collect_duration
        elif hasattr(self.meta, "Global"):
            try:
                collect_start = self.meta.Global.CollectStart
            except Exception as e:
                logging.error("No field found in Global.CollectStart.  {}".format(e))
        # TODO: implement collect duration for the case where vbmeta.TxTime is used.
            return collect_start, None
        else:
            return None, None

    def get_polarization(self):
        pol = None
        if hasattr(self.meta, "ImageFormation"):
            try:
                pol = self.meta.ImageFormation.TxRcvPolarizationProc
            except Exception as e:
                logging.error("Polarization not found {}".format(e))
        elif hasattr(self.meta, "RadarCollection"):
            try:
                pol = self.meta.RadarCollection.TxPolarization
            except Exception as e:
                logging.error("Polarization not found {}".format(e))
        return pol

    def get_geo_lon_lat(self):
        if hasattr(self.meta, "GeoData"):
            try:
                scp = [self.meta.GeoData.SCP.ECF.X, self.meta.GeoData.SCP.ECF.Y, self.meta.GeoData.SCP.ECF.Z]
            except Exception as e:
                logging.error("Unable to get geolocation data in ECF form {}".format(e))
        # TODO: might take this out if it's not part of the SICD standard
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
        shadow = self.meta.SCPCOA.Shadow
        multipath = self.meta.SCPCOA.Multipath
        azimuth = self.meta.SCPCOA.AzimAng
        layover = self.meta.SCPCOA.LayoverAng
        if hasattr(self.meta, "Grid") or self.meta.Grid.ImagePlane == 'SLANT':
            shadow = azimuth - 180 - self.meta.SCPCOA.MultipathGround
            multipath = azimuth - 180
            layover = layover - self.meta.SCPCOA.MultipathGround

        layover = 90 - (layover - azimuth)
        shadow = 90 - (shadow - azimuth)
        north = azimuth + 90
        multipath = north - multipath
        return layover, shadow, multipath, north

    def draw_arrow(self,
                   arrow_type,  # type: str
                   arrow_width=2,  # type: int
                   ):
        layover, shadow, multipath, north = self.get_arrow_layover_shadow_multipath_north_angles()

        if arrow_type.lower() == "layover":
            arrow = layover
            arrow_color = self.layover_color
        elif arrow_type.lower() == "shadow":
            arrow = shadow
            arrow_color = self.shadow_color
        elif arrow_type.lower() == "multipath":
            arrow = multipath
            arrow_color = self.multipath_color
        elif arrow_type.lower() == "north":
            arrow = north
            arrow_color = self.north_color

        arrow_rad = numpy.deg2rad(arrow)

        arrow_length_old = self.canvas.canvas_width * 0.15
        arrows_origin = (self.canvas.canvas_width * 0.75, self.canvas.canvas_height * 0.6)

        # adjust aspect ratio in the case we're dealing with circular polarization from RCM
        pixel_aspect_ratio = 1.0
        aspect_ratio = 1.0
        if hasattr(self.meta, "Grid") and \
                hasattr(self.meta.Grid, "Col") and \
                hasattr(self.meta.Grid, "Row") and \
                hasattr(self.meta.Grid.Col, "SS") and \
                hasattr(self.meta.Grid.Row, "SS"):
            pixel_aspect_ratio = self.meta.Grid.Col.SS / self.meta.Grid.Row.SS
            aspect_ratio = aspect_ratio * pixel_aspect_ratio

        if aspect_ratio > 1:
            new_length = numpy.sqrt(numpy.square(arrow_length_old * numpy.cos(arrow_rad) / aspect_ratio) +
                                 numpy.square(arrow_length_old * numpy.sin(arrow_rad)))
            arrow_length = arrow_length_old * arrow_length_old / new_length
            x_end = arrows_origin[0] + arrow_length * numpy.cos(arrow_rad) / aspect_ratio
            y_end = arrows_origin[1] - arrow_length * numpy.sin(arrow_rad)
        else:
            new_length = numpy.sqrt(numpy.square(arrow_length_old * numpy.cos(arrow_rad)) +
                                 numpy.square(arrow_length_old * numpy.sin(arrow_rad) * aspect_ratio))
            arrow_length = arrow_length_old * arrow_length_old / new_length
            x_end = arrows_origin[0] + arrow_length * numpy.cos(arrow_rad)
            y_end = arrows_origin[1] - arrow_length * numpy.sin(arrow_rad) * aspect_ratio

        # now draw the arrows
        self.canvas.create_new_arrow((arrows_origin[0],
                               arrows_origin[1],
                               x_end,
                               y_end),
                              fill=arrow_color,
                              width=arrow_width)

        # label the north arrow
        if arrow_type.lower() == "north":
            line_positions = self._get_line_positions()
            text_height = int((line_positions[1][1] - line_positions[0][1]) * 0.7)
            canvas_font = font.Font(family='Times New Roman', size=-text_height)

            self.canvas.create_text(x_end + (x_end - arrows_origin[0]) * 0.2,
                                    y_end + (y_end - arrows_origin[1]) * 0.2,
                                    text="N",
                                    fill=self.north_color,
                                    font=canvas_font)
