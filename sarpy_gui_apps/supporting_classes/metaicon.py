import sarpy.io.complex as sarpy_complex
import datetime
from sarpy.geometry import geocoords
import numpy as np


class MetaIcon:
    __slots__ = ('fname', 'reader_object')

    def __init__(self, fname):
        self.fname = fname
        self.reader_object = sarpy_complex.open(fname)

    def get_all_meta_icon_data(self):
        collect_start, collect_curation = self.get_timings()
        iid_line = self.get_iid_line()
        geo_line = self.get_geo_line()

    def get_iid_line(self):
        meta = self.reader_object.sicd_meta
        if hasattr(meta, "Timeline")and hasattr(meta.Timeline, "CollectStart") and hasattr(meta, "CollectionInfo") and hasattr(meta.CollectionInfo, "CollectorName"):
            date_str = meta.Timeline.CollectStart.astype(datetime.datetime).strftime("%d%b%y").upper()
            collector_name_str = meta.CollectionInfo.CollectorName
            if len(collector_name_str) > 4:
                collector_name_str = collector_name_str[0:4]
            iid_line = date_str + " " + collector_name_str
        elif hasattr(meta, "Global") and hasattr(meta.Global, "CoreName") and hasattr(meta, "collectionInfo"):
            iid_line = meta.CollectionInfo.CoreName
            if len(iid_line) > 16:
                iid_line = iid_line[0:16]
        else:
            iid_line = ""
        if hasattr(meta.Timeline, "CollectStart"):
            iid_datestr = meta.Timeline.CollectStart.astype(datetime.datetime).strftime("%H%MZ")
            iid_line = iid_line + " / " + iid_datestr
        return iid_line

    def get_geo_line(self):
        lat, lon = self.get_geo_lon_lat()
        geo_line = "Geo: " + "{:.4f}".format(lat) + "/" + "{:.4f}".format(lon)
        return geo_line

    def get_timings(self):
        meta = self.reader_object.sicd_meta
        if hasattr(meta, "Timeline"):
            if hasattr(meta.Timeline, "CollectStart"):
                collect_start = meta.Timeline.CollectStart
            if hasattr(meta.Timeline, "CollectDuration"):
                collect_duration = meta.Timeline.CollectDuration
        elif hasattr(meta, "Global"):
            if hasattr(meta.Global, "CollectStart"):
                collect_start = meta.Global.CollectStart
        # TODO: implement collect duration for the case where vbmeta.TxTime is used.  Matlab code below:
        # elseif
        # exist('vbmeta', 'var')
        # CollectDuration = vbmeta.TxTime(end);
        return collect_start, collect_duration

    # TODO
    def get_country_code(self):
        pass

    def get_geo_lon_lat(self):
        meta = self.reader_object.sicd_meta
        if hasattr(meta, "GeoData") and hasattr(meta.GeoData, "SCP") and hasattr(meta.GeoData.SCP, "ECF"):
            scp = [meta.GeoData.SCP.ECF.X, meta.GeoData.SCP.ECF.Y, meta.GeoData.SCP.ECF.Z]
        elif hasattr(meta, "SRP"):
            if meta.SRP.SRPType == "FIXEDPT":
                scp = meta.SRP.FIXEDPT.SRPPT
                scp = [scp.X, scp.Y, scp.Z]
            elif meta.SRP.SRPType == "PVTPOLY":
                # TODO: implement this for the case where we need to do a polynomial
                pass
        # TODO: implement vbmeta version

        # TODO: make this match the matlab version, if that's important.
        if scp:
            lla = geocoords.ecf_to_geodetic(scp)
            lat = lla[0]
            lon = lla[1]
        return lon, lat
