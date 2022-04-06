# MIT License
#
# Copyright (c) 2020 National Geospatial-Intelligence Agency
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


__all__ = ['__version__',
           '__classification__', '__author__', '__url__', '__email__',
           '__title__', '__summary__',
           '__license__', '__copyright__']

from sarpy.__details__ import __classification__, _post_identifier
_version_number = '1.2.66'

__version__ = _version_number + _post_identifier

__author__ = "National Geospatial-Intelligence Agency"
__url__ = "https://github.com/ngageoint/sarpy"
__email__ = "Wade.C.Schwartzkopf@nga.mil"


__title__ = "sarpy"
__summary__ = "Python tools for reading, writing, and simple processing of complex SAR data and other " \
              "associated data."


__license__ = "MIT License"
__copyright__ = "2020 {}".format(__author__)
