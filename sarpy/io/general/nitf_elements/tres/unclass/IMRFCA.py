# -*- coding: utf-8 -*-

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class XINC(TREElement):
    def __init__(self, value):
        super(XINC, self).__init__()
        self.add_field('XINC', 's', 22, value)


class XIDC(TREElement):
    def __init__(self, value):
        super(XIDC, self).__init__()
        self.add_field('XIDC', 's', 22, value)


class YINC(TREElement):
    def __init__(self, value):
        super(YINC, self).__init__()
        self.add_field('YINC', 's', 22, value)


class YIDC(TREElement):
    def __init__(self, value):
        super(YIDC, self).__init__()
        self.add_field('YIDC', 's', 22, value)

class IMRFCAType(TREElement):
    def __init__(self, value):
        super(IMRFCAType, self).__init__()
        self.add_loop('XINCs', 20, XINC, value)
        self.add_loop('XIDCs', 20, XIDC, value)
        self.add_loop('YINCs', 20, YINC, value)
        self.add_loop('YIDCs', 20, YIDC, value)


class IMRFCA(TREExtension):
    _tag_value = 'IMRFCA'
    _data_type = IMRFCAType
