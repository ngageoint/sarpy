
from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class SITE(TREElement):
    def __init__(self, value):
        super(SITE, self).__init__()
        self.add_field('SCLCTN_NAME', 's', 20, value)
        self.add_field('SDESCRIPTION', 's', 255, value)
        self.add_field('SITE_NUM', 's', 3, value)
        self.add_field('SCN_NUM', 's', 3, value)
        self.add_field('SCLCTN_STDATE', 's', 8, value)
        self.add_field('SCLCTN_SPDATE', 's', 8, value)
        self.add_field('SCN_CNTR', 's', 11, value)
        self.add_field('ALTITUDE', 's', 5, value)
        self.add_field('SCN_CONTENT', 's', 50, value)
        self.add_field('BGRND_TYPE', 's', 50, value)
        self.add_field('SITE_COV', 's', 1, value)


class CLCTNBType(TREElement):
    def __init__(self, value):
        super(CLCTNBType, self).__init__()
        self.add_field('VERNUM', 's', 4, value)
        self.add_field('CLCTN_NAME', 's', 25, value)
        self.add_field('CLCTN_DESCRIPT', 's', 255, value)
        self.add_field('CLCTN_STDATE', 's', 8, value)
        self.add_field('CLCTN_SPDATE', 's', 8, value)
        self.add_field('CLCTN_LOC', 's', 11, value)
        self.add_field('SITE', 's', 15, value)
        self.add_field('COUNTRY', 's', 2, value)
        self.add_field('SPONSOR', 's', 20, value)
        self.add_field('PERSONNEL', 's', 100, value)
        self.add_field('NUM_SITES', 'd', 1, value)
        self.add_loop('SITEs', self.NUM_SITES, SITE, value)
        self.add_field('SCLCTN_Z_OFF', 's', 3, value)
        self.add_field('SECURITY', 's', 7, value)
        self.add_field('SCG', 's', 15, value)
        self.add_field('FLIGHT_NUM', 's', 2, value)
        self.add_field('PASS_NUM', 's', 2, value)
        self.add_field('WX_STATION', 's', 20, value)
        self.add_field('WX_OVERVIEW', 's', 15, value)
        self.add_field('WX_FILE', 's', 30, value)


class CLCTNB(TREExtension):
    _tag_value = 'CLCTNB'
    _data_type = CLCTNBType
