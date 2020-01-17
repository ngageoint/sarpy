#!/usr/bin/python

# This file is mostly a straight translation of
# GeographicLib/src/Geoid.cpp from C++ to Python
# by Kim Vandry <vandry@TZoNE.ORG>
#
# /**
# * \file Geoid.cpp
# * \brief Implementation for GeographicLib::Geoid class
# *
# * Copyright (c) Charles Karney (2009) <charles@karney.com>
# * and licensed under the LGPL.  For more information, see
# * http://geographiclib.sourceforge.net/
# **********************************************************************/
#
# Geoid height grade not supported

import os
import mmap
import struct
import platform

class GeoidBadDataFile(Exception):
    pass

class GeoidHeight(object):
    """Calculate the height of the WGS84 geoid above the
    ellipsoid at any given latitude and longitude
    :param name: name to PGM file containing model info
    download from http://geographiclib.sourceforge.net/1.18/geoid.html
    """
    c0 = 240
    c3 = (
        (  9, -18, -88,    0,  96,   90,   0,   0, -60, -20),
        ( -9,  18,   8,    0, -96,   30,   0,   0,  60, -20),
        (  9, -88, -18,   90,  96,    0, -20, -60,   0,   0),
        (186, -42, -42, -150, -96, -150,  60,  60,  60,  60),
        ( 54, 162, -78,   30, -24,  -90, -60,  60, -60,  60),
        ( -9, -32,  18,   30,  24,    0,  20, -60,   0,   0),
        ( -9,   8,  18,   30, -96,    0, -20,  60,   0,   0),
        ( 54, -78, 162,  -90, -24,   30,  60, -60,  60, -60),
        (-54,  78,  78,   90, 144,   90, -60, -60, -60, -60),
        (  9,  -8, -18,  -30, -24,    0,  20,  60,   0,   0),
        ( -9,  18, -32,    0,  24,   30,   0,   0, -60,  20),
        (  9, -18,  -8,    0, -24,  -30,   0,   0,  60,  20),
    )

    c0n = 372
    c3n = (
        (  0, 0, -131, 0,  138,  144, 0,   0, -102, -31),
        (  0, 0,    7, 0, -138,   42, 0,   0,  102, -31),
        ( 62, 0,  -31, 0,    0,  -62, 0,   0,    0,  31),
        (124, 0,  -62, 0,    0, -124, 0,   0,    0,  62),
        (124, 0,  -62, 0,    0, -124, 0,   0,    0,  62),
        ( 62, 0,  -31, 0,    0,  -62, 0,   0,    0,  31),
        (  0, 0,   45, 0, -183,   -9, 0,  93,   18,   0),
        (  0, 0,  216, 0,   33,   87, 0, -93,   12, -93),
        (  0, 0,  156, 0,  153,   99, 0, -93,  -12, -93),
        (  0, 0,  -45, 0,   -3,    9, 0,  93,  -18,   0),
        (  0, 0,  -55, 0,   48,   42, 0,   0,  -84,  31),
        (  0, 0,   -7, 0,  -48,  -42, 0,   0,   84,  31),
    )

    c0s = 372
    c3s = (
        ( 18,  -36, -122,   0,  120,  135, 0,   0,  -84, -31),
        (-18,   36,   -2,   0, -120,   51, 0,   0,   84, -31),
        ( 36, -165,  -27,  93,  147,   -9, 0, -93,   18,   0),
        (210,   45, -111, -93,  -57, -192, 0,  93,   12,  93),
        (162,  141,  -75, -93, -129, -180, 0,  93,  -12,  93),
        (-36,  -21,   27,  93,   39,    9, 0, -93,  -18,   0),
        (  0,    0,   62,   0,    0,   31, 0,   0,    0, -31),
        (  0,    0,  124,   0,    0,   62, 0,   0,    0, -62),
        (  0,    0,  124,   0,    0,   62, 0,   0,    0, -62),
        (  0,    0,   62,   0,    0,   31, 0,   0,    0, -31),
        (-18,   36,  -64,   0,   66,   51, 0,   0, -102,  31),
        ( 18,  -36,    2,   0,  -66,  -51, 0,   0,  102,  31),
    )

    def __init__(self, name="egm2008-1.pgm"):
        self.offset = None
        self.scale = None

        with open(name, "rb") as f:
            line = f.readline()
            if line != b"P5\012" and line != b"P5\015\012":
                raise GeoidBadDataFile("No PGM header")
            headerlen = len(line)
            while True:
                line = f.readline()
                if len(line) == 0:
                    raise GeoidBadDataFile("EOF before end of file header")
                headerlen += len(line)
                if line.startswith(b'# Offset '):
                    try:
                        self.offset = int(line[9:])
                    except ValueError as e:
                        raise GeoidBadDataFile("Error reading offset", e)
                elif line.startswith(b'# Scale '):
                    try:
                        self.scale = float(line[8:])
                    except ValueError as e:
                        raise GeoidBadDataFile("Error reading scale", e)
                elif not line.startswith(b'#'):
                    try:
                        self.width, self.height = list(map(int, line.split()))
                    except ValueError as e:
                        raise GeoidBadDataFile("Bad PGM width&height line", e)
                    break
            line = f.readline()
            headerlen += len(line)
            levels = int(line)
            if levels != 65535:
                raise GeoidBadDataFile("PGM file must have 65535 gray levels")
            if self.offset is None:
                raise GeoidBadDataFile("PGM file does not contain offset")
            if self.scale is None:
                raise GeoidBadDataFile("PGM file does not contain scale")

            if self.width < 2 or self.height < 2:
                raise GeoidBadDataFile("Raster size too small")

            fd = f.fileno()
            fullsize = os.fstat(fd).st_size

            if fullsize - headerlen != self.width * self.height * 2:
                raise GeoidBadDataFile("File has the wrong length")

            self.headerlen = headerlen
            osplat = platform.system()
            if(osplat == "Linux"):
                self.raw = mmap.mmap(fd, fullsize, mmap.MAP_SHARED, mmap.PROT_READ)
            elif(osplat == "Windows"):
                self.raw = mmap.mmap(fd, fullsize, access=mmap.ACCESS_READ)
            else:
                self.raw = mmap.mmap(fd, fullsize, mmap.MAP_SHARED, mmap.PROT_READ)
        self.rlonres = self.width / 360.0
        self.rlatres = (self.height - 1) / 180.0
        self.ix = None
        self.iy = None

    def _rawval(self, ix, iy):
        if iy < 0:
            iy = -iy;
            ix += self.width/2;
        elif iy >= self.height:
            iy = 2 * (self.height - 1) - iy;
            ix += self.width/2;
        if ix < 0:
            ix += self.width;
        elif ix >= self.width:
            ix -= self.width

        return struct.unpack_from('>H', self.raw,
                (iy * self.width + ix) * 2 + self.headerlen
        )[0]

    def get(self, lat, lon, cubic=True):
        if lon < 0:
            lon += 360
        fy = (90 - lat) * self.rlatres
        fx = lon * self.rlonres
        iy = int(fy)
        ix = int(fx)
        fx -= ix
        fy -= iy
        if iy == self.height - 1:
            iy -= 1

        if ix != self.ix or iy != self.iy:
            self.ix = ix
            self.iy = iy
            if not cubic:
                self.v00 = self._rawval(ix, iy)
                self.v01 = self._rawval(ix+1, iy)
                self.v10 = self._rawval(ix, iy+1)
                self.v11 = self._rawval(ix+1, iy+1)
            else:
                v = (
                    self._rawval(ix    , iy - 1),
                    self._rawval(ix + 1, iy - 1),
                    self._rawval(ix - 1, iy    ),
                    self._rawval(ix    , iy    ),
                    self._rawval(ix + 1, iy    ),
                    self._rawval(ix + 2, iy    ),
                    self._rawval(ix - 1, iy + 1),
                    self._rawval(ix    , iy + 1),
                    self._rawval(ix + 1, iy + 1),
                    self._rawval(ix + 2, iy + 1),
                    self._rawval(ix    , iy + 2),
                    self._rawval(ix + 1, iy + 2)
                )
                if iy == 0:
                    c3x = GeoidHeight.c3n
                    c0x = GeoidHeight.c0n
                elif iy == self.height - 2:
                    c3x = GeoidHeight.c3s
                    c0x = GeoidHeight.c0s
                else:
                    c3x = GeoidHeight.c3
                    c0x = GeoidHeight.c0
                self.t = [
                    sum([ v[j] * c3x[j][i] for j in range(12) ]) / float(c0x)
                    for i in range(10)
                ]
        if not cubic:
            a = (1 - fx) * self.v00 + fx * self.v01
            b = (1 - fx) * self.v10 + fx * self.v11
            h = (1 - fy) * a + fy * b
        else:
            h = (
                self.t[0] +
                fx * (self.t[1] + fx * (self.t[3] + fx * self.t[6])) +
                fy * (
                    self.t[2] + fx * (self.t[4] + fx * self.t[7]) +
                        fy * (self.t[5] + fx * self.t[8] + fy * self.t[9])
                )
            )
        return self.offset + self.scale * h