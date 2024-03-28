"""
Remove the existing spectral taper window from a sicd-type, if necessary,
then apply a new spectral taper window.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Valkyrie Systems Corporation"

import copy

import numpy as np
import scipy.fft
import scipy.interpolate as spi

from sarpy.io.complex.sicd_elements.Grid import WgtTypeType
from sarpy.processing.sicd.normalize_sicd import apply_skew_poly
import sarpy.processing.sicd.windows as windows


class Taper:
    """
    This is a helper class that wraps various window function in a common interface.

    Args
    ----
    window: str
        The name of the taper (window) function name.  Acceptable (case insensitive) names are:

        * "uniform": This is a flat spectral taper (i.e., no spectral taper).
        * "hamming": Hamming window taper (0.54 + 0.46 * cos(2*pi*n/M))
        * "hanning" | "hann":  Hann window taper (0.5 + 0.5 * cos(2*pi*n/M))
        * "general_hamming": Raised cosine window (alpha + (1 - alpha) * cos(2*pi*n/M)), default (alpha = 0.5)
        * "kaiser": Kaiser-Bessel window (I0[beta * sqrt(1 - (2*n/M - 1)**2)] / I0[beta]) default (beta = 4)
        * "taylor": Taylor window: Default is a nbar = 4, sll = -30.

    pars: dict | None
        Optional dictionary of parameter values for those windows that have parameter values.
        If omitted or None, the default parameters are used.
        The parameter values for each window are listed above.

        * "uniform": no parameters
        * "hamming": no parameters
        * "hanning" | "hann": no parameters
        * "general_hamming": parameter "alpha"
                 Raised cosine window, w[n] = (alpha + (1 - alpha) * cos(2*pi*n/M)).
                 The default value is alpha = 0.5
        * "kaiser": parameter "beta"
                Kaiser-Bessel window, w[n] = I0[beta * sqrt(1 - (2*n/M - 1)**2)] / I0[beta]
                The default value is beta = 4.
        * "taylor": parameters "nbar" and "sll"
                "nbar" = number of near side lobes and "sll" = side lobe level (dB) of near side lobes.
                The default values are nbar = 4, sll = -30.

    default_size: int
        Optional taper window size.  This argument is almost never needed (see note below).  (default: 513)

    Note
    ----
    This class always creates a symmetric window regardless of whether the window length is odd or even.
    It is not intended that the internal representation of the window samples be used directly.  Instead,
    the "get_vals(size, sym)" method should be used.  Accessing the window values this way will give you
    full control over the size of the window function and whether the window is symmetric.

    """
    def __init__(self, window='UNIFORM', pars=None, default_size=513):
        pars = {key.upper(): val for key, val in pars.items()} if pars else {}    # Force upper-case parameter names

        self.default_pars = {"UNIFORM": {},
                             "HAMMING": {},
                             "HANNING": {},
                             "HANN": {},
                             "GENERAL_HAMMING": {'ALPHA': 0.50},
                             "KAISER": {'BETA': 4.0},
                             "TAYLOR": {"NBAR": 4, "SLL": -30.0},
                             }
        self.default_size = default_size
        self.window_type = window.upper()
        self.window_pars = {**self.default_pars.get(self.window_type, {}), **pars}
        self.window_vals = self._make_sym_1d_window(self.default_size)

    def _make_sym_1d_window(self, window_size=None):
        """Make the sample values of a 1-D symmetric window"""
        window_size = self.default_size if window_size is None else window_size

        if self.window_type == 'UNIFORM':
            wgts = np.ones(window_size)

        elif self.window_type in ['HAMMING']:
            wgts = windows.hamming(window_size, sym=True)

        elif self.window_type in ['HANNING', 'HANN']:
            wgts = windows.hanning(window_size, sym=True)

        elif self.window_type in ['GENERAL_HAMMING']:
            alpha = float(self.window_pars['ALPHA'])
            wgts = windows.general_hamming(window_size, alpha=alpha, sym=True)

        elif self.window_type == 'TAYLOR':
            nbar = int(self.window_pars['NBAR'])
            sll = float(self.window_pars['SLL'])
            wgts = windows.taylor(window_size, nbar=nbar, sll=-np.abs(sll), sym=True)

        elif self.window_type == 'KAISER':
            beta = float(self.window_pars['BETA'])
            wgts = windows.kaiser(window_size, beta=beta, sym=True)

        else:
            raise ValueError(f'Window type "{self.window_type}" is not supported.')

        return wgts

    def get_vals(self, size, sym=True):
        """
        Get interpolated samples of the prototype window.

        Args
        ----
        size: int
            The number of output samples
        sym: bool
            Whether interpolated samples should be symmetric (Default: True)

        Returns
        -------
        vals: numpy.ndarray
            1-D array of sampled taper values.
        """
        return _fit_1d_window(self.window_vals, size) if sym else _fit_1d_window(self.window_vals, size+1)[:-1]


def apply_spectral_taper(sicd_reader, taper):
    """
    Apply a spectral taper window function to SICD image data.
    If necessary, the existing spectral taper window will be removed prior to applying
    the new spectral taper window. To remove an existing spectral taper window without
    applying a new taper, specify a new taper=None.

    Args
    ----
    sicd_reader: sarpy.io.complex.sicd.SICDReader
        A SICD reader object containing both image data and metadata.

    taper: Taper | str | None
        A Taper object specifying the spectral domain taper function to be applied.
        Alternative, a string indicating the taper type.  When a string is provided, a Taper
        object will be created with default parameters for the specified taper type.  Use a
        Taper object if you need to modify the taper parameters.  For example::

            apply_spectral_taper(sicd, Taper(window_name, pars=pars_dict))

    Returns
    -------
    cdata: numpy.ndarray
        The sicd image data modified by removing the existing spectral taper window and applying the specified window.

    sicd_mdata: SICDType
        The modified sicd metadata with updated Grid parameters

    """
    taper = "UNIFORM" if taper is None else taper

    if isinstance(taper, str):
        taper = Taper(taper)

    cdata = sicd_reader[:, :]
    mdata = copy.deepcopy(sicd_reader.sicd_meta)

    cdata = _apply_2d_spectral_taper(cdata, mdata, taper.window_vals)

    # Update the SICD metadata to account for the spectral weighting changes
    row_inv_osf = mdata.Grid.Row.ImpRespBW * mdata.Grid.Row.SS
    col_inv_osf = mdata.Grid.Col.ImpRespBW * mdata.Grid.Col.SS

    old_row_wgts = _get_sicd_wgt_funct(mdata, 'Row', len(taper.window_vals))
    old_col_wgts = _get_sicd_wgt_funct(mdata, 'Col', len(taper.window_vals))
    old_coh_amp_gain = np.mean(old_row_wgts) * np.mean(old_col_wgts) * row_inv_osf * col_inv_osf
    old_rms_amp_gain = np.sqrt(np.mean(old_row_wgts**2) * np.mean(old_col_wgts**2) * row_inv_osf * col_inv_osf)

    new_row_wgts = taper.window_vals
    new_col_wgts = taper.window_vals
    new_coh_amp_gain = np.mean(new_row_wgts) * np.mean(new_col_wgts) * row_inv_osf * col_inv_osf
    new_rms_amp_gain = np.sqrt(np.mean(new_row_wgts**2) * np.mean(new_col_wgts**2) * row_inv_osf * col_inv_osf)

    coh_pwr_gain = (new_coh_amp_gain / old_coh_amp_gain) ** 2
    rms_pwr_gain = (new_rms_amp_gain / old_rms_amp_gain) ** 2

    mdata.Grid.Row.WgtType = WgtTypeType(WindowName=taper.window_type.upper(), Parameters=taper.window_pars)
    mdata.Grid.Col.WgtType = WgtTypeType(WindowName=taper.window_type.upper(), Parameters=taper.window_pars)

    taper_is_uniform = np.all(taper.window_vals == taper.window_vals[0])
    mdata.Grid.Row.WgtFunct = None if taper_is_uniform else taper.window_vals
    mdata.Grid.Col.WgtFunct = None if taper_is_uniform else taper.window_vals

    ipr_half_power_width = windows.find_half_power(taper.window_vals, oversample=16)
    mdata.Grid.Row.ImpRespWid = ipr_half_power_width / mdata.Grid.Row.ImpRespBW
    mdata.Grid.Col.ImpRespWid = ipr_half_power_width / mdata.Grid.Col.ImpRespBW

    if mdata.Radiometric:
        if mdata.Radiometric.NoiseLevel:
            if mdata.Radiometric.NoiseLevel.NoiseLevelType == "ABSOLUTE":
                mdata.Radiometric.NoiseLevel.NoisePoly.Coefs *= rms_pwr_gain

        if mdata.Radiometric.RCSSFPoly:
            mdata.Radiometric.RCSSFPoly.Coefs /= coh_pwr_gain

        if mdata.Radiometric.SigmaZeroSFPoly:
            mdata.Radiometric.SigmaZeroSFPoly.Coefs /= rms_pwr_gain

        if mdata.Radiometric.BetaZeroSFPoly:
            mdata.Radiometric.BetaZeroSFPoly.Coefs /= rms_pwr_gain

        if mdata.Radiometric.GammaZeroSFPoly:
            mdata.Radiometric.GammaZeroSFPoly.Coefs /= rms_pwr_gain

    return cdata, mdata


def _apply_2d_spectral_taper(cdata, mdata, window_vals):
    for axis in ['Row', 'Col']:
        existing_window = _get_sicd_wgt_funct(mdata, axis, len(window_vals))
        both_windows = window_vals / np.maximum(existing_window, 0.01 * np.max(existing_window))

        cdata = _apply_1d_spectral_taper(cdata, mdata, axis, both_windows)

    return cdata


def _apply_1d_spectral_taper(cdata, mdata, axis, window_vals):
    axis_index = {'Row': 0, 'Col': 1}[axis]
    axis_mdata = {'Row': mdata.Grid.Row, 'Col': mdata.Grid.Col}[axis]

    xrow = (np.arange(0, mdata.ImageData.NumRows) - mdata.ImageData.SCPPixel.Row) * mdata.Grid.Row.SS
    ycol = (np.arange(0, mdata.ImageData.NumCols) - mdata.ImageData.SCPPixel.Col) * mdata.Grid.Col.SS

    delta_k_coa_poly = np.array([[0.0]]) if axis_mdata.DeltaKCOAPoly is None else axis_mdata.DeltaKCOAPoly.Coefs

    if not np.all(delta_k_coa_poly == 0):
        cdata = apply_skew_poly(cdata, delta_k_coa_poly, row_array=xrow, col_array=ycol,
                                fft_sgn=axis_mdata.Sgn, dimension=axis_index, forward=False)

    cdata = _fft_window_ifft(cdata, mdata, axis, window_vals)

    if not np.all(delta_k_coa_poly == 0):
        cdata = apply_skew_poly(cdata, delta_k_coa_poly, row_array=xrow, col_array=ycol,
                                fft_sgn=axis_mdata.Sgn, dimension=axis_index, forward=True)
    return cdata


def _get_sicd_wgt_funct(mdata, axis, desired_size=513):
    axis_mdata = {'Row': mdata.Grid.Row, 'Col': mdata.Grid.Col}[axis]

    if axis_mdata.WgtFunct is not None:
        # Get the SICD WgtFunct values and interpolate as needed to produce a window of the desired size.
        wgts = _fit_1d_window(axis_mdata.WgtFunct, desired_size=desired_size)

    else:
        # The SICD WgtFunct values do not exist, so if WindowName is not specified,
        # or the WindowName is "UNIFORM", then return a window of all ones.
        # If WindowName is specified as other than "UNIFORM" then we can not presume to
        # know what the WindowName means because it is not clearly defined in the SICD spec.
        window_name = axis_mdata.WgtType.WindowName if axis_mdata.WgtType else "UNIFORM"

        if window_name.upper() == 'UNIFORM':
            wgts = np.ones(desired_size)
        else:
            raise ValueError(f'SICD/Grid/{axis}/WgtFunct is not part of the SICD metadata, but there appears '
                             f'to be a window of type "{window_name}" applied to the {axis} axis spectrum.')

    return wgts


def _fit_1d_window(window_vals, desired_size):
    if len(window_vals) == desired_size:
        w = window_vals
    else:
        x = np.linspace(0, 1, len(window_vals))
        f = spi.interp1d(x, window_vals, kind='cubic')
        w = f(np.linspace(0, 1, desired_size))
    return w


def _fft_window_ifft(cdata, mdata, axis, window_vals):
    axis_index = {'Row': 0, 'Col': 1}[axis]
    axis_mdata = {'Row': mdata.Grid.Row, 'Col': mdata.Grid.Col}[axis]
    nyquist_bw = 1.0 / axis_mdata.SS
    ipr_bw = axis_mdata.ImpRespBW
    osf = nyquist_bw / ipr_bw

    # Zero pad the image data to avoid IPR wrap-around, then
    # find a good FFT size which creates some additional zero pad.
    axis_size = cdata.shape[axis_index]
    wrap_around_pad = int(min(200 * osf, 0.1 * axis_size))
    good_fft_size = scipy.fft.next_fast_len(axis_size + wrap_around_pad)

    # Forward transform without FFTSHIFT so the DC bin is at index=0
    if axis_mdata.Sgn == -1:
        cdata_fft = scipy.fft.fft(cdata, n=good_fft_size, axis=axis_index)
    else:
        cdata_fft = scipy.fft.ifft(cdata, n=good_fft_size, axis=axis_index)

    # Interpolate the taper to cover the spectral support bandwidth and extend the
    # taper window's end points into the over sample region of the spectrum.
    f = spi.interp1d(np.linspace(-ipr_bw / 2, ipr_bw / 2, len(window_vals)), window_vals, kind='cubic',
                     bounds_error=False, fill_value=(window_vals[0], window_vals[-1]))
    padded_taper = f(scipy.fft.fftfreq(good_fft_size, axis_mdata.SS))

    # Apply the taper to the spectrum.
    taper_2d = padded_taper[:, np.newaxis] if axis == 'Row' else padded_taper[np.newaxis, :]
    cdata_fft *= taper_2d

    # Inverse transform without FFTSHIFT and trim back to the original image size.
    nrows, ncols = cdata.shape
    if axis_mdata.Sgn == -1:
        cdata = scipy.fft.ifft(cdata_fft, n=good_fft_size, axis=axis_index)[:nrows, :ncols]
    else:
        cdata = scipy.fft.fft(cdata_fft, n=good_fft_size, axis=axis_index)[:nrows, :ncols]

    return cdata
