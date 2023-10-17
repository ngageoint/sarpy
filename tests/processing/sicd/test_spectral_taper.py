"""
These test functions will exercise the spectral_taper module which contains functions used to
apply/remove spectral weighting functions to SICD data.

"""
import copy
import logging

import numpy as np
import pytest
import scipy
import scipy.fft as scifft
import scipy.signal.windows as sciwin

from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.complex.sicd_elements.ImageData import ImageDataType
from sarpy.io.complex.sicd_elements.Grid import GridType, DirParamType, WgtTypeType
from sarpy.io.complex.sicd_elements.Radiometric import RadiometricType, NoiseLevelType_, Poly2DType

import sarpy.processing.sicd.windows as windows
import sarpy.processing.sicd.spectral_taper as spectral_taper

logging.basicConfig(level=logging.WARNING)

old_scipy = [int(d) for d in scipy.__version__.split('.')][:2] < [1, 6]


@pytest.fixture()
def mock_sicd_meta():
    nrows = 128
    ncols = 256

    row_ss = 1.5
    col_ss = 1.5
    ipr_wid = row_ss * 1.25
    ipr_bw = 0.886 / ipr_wid

    window_size = 17
    alpha = 0.68
    wgts = alpha + (1 - alpha) * np.cos(np.linspace(-np.pi, np.pi, window_size))

    sicd_imagedata = ImageDataType(PixelType="RE32F_IM32F",
                                   NumRows=nrows, NumCols=ncols,
                                   FirstRow=0, FirstCol=0,
                                   SCPPixel=(nrows//2, ncols//2)
                                   )

    grid_row = DirParamType(UVectECF=(1, 0, 0), SS=row_ss, ImpRespWid=ipr_wid, Sgn=-1, ImpRespBW=ipr_bw,
                            KCtr=0, DeltaK1=-ipr_bw/2, DeltaK2=ipr_bw/2, DeltaKCOAPoly=np.array([[0.0]]),
                            WgtType=WgtTypeType(WindowName="Hamming", Parameters={"ALPHA": alpha}), WgtFunct=wgts)
    grid_col = DirParamType(UVectECF=(0, 1, 0), SS=col_ss, ImpRespWid=ipr_wid, Sgn=-1, ImpRespBW=ipr_bw,
                            KCtr=0, DeltaK1=-ipr_bw/2, DeltaK2=ipr_bw/2, DeltaKCOAPoly=np.array([[0.0]]),
                            WgtType=WgtTypeType(WindowName="Hamming", Parameters={"ALPHA": alpha}), WgtFunct=wgts)
    sicd_grid = GridType(ImagePlane="SLANT", Type="RGAZIM", Row=grid_row, Col=grid_col)

    sicd_radiometric = RadiometricType(NoiseLevel=NoiseLevelType_(NoiseLevelType="ABSOLUTE",
                                                                  NoisePoly=Poly2DType([[1.0]])),
                                       RCSSFPoly=Poly2DType([[1.0]]),
                                       SigmaZeroSFPoly=Poly2DType([[1.0]]),
                                       BetaZeroSFPoly=Poly2DType([[1.0]]),
                                       GammaZeroSFPoly=Poly2DType([[1.0]]))

    mdata = SICDType(ImageData=sicd_imagedata, Grid=sicd_grid, Radiometric=sicd_radiometric)

    return mdata


@pytest.mark.parametrize("window,parlist", [("uniform", None),
                                            ("hamming", None),
                                            ("hanning", None),
                                            ("hann", None),
                                            ('general_hamming', None),
                                            ("kaiser", None),
                                            ('general_hamming', ("alpha", 0.6)),
                                            ("kaiser", ("beta", 15.0)),
                                            pytest.param("taylor", None,
                                                         marks=pytest.mark.skipif(old_scipy, reason="scipy ver < 1.6")),
                                            pytest.param('taylor', ("nbar", 5, "sll", -35.0),
                                                         marks=pytest.mark.skipif(old_scipy, reason="scipy ver < 1.6"))
                                            ])
def test_make_sym_1d_window(window, parlist):
    n_evn = 20
    n_odd = n_evn + 1

    pars = None
    if parlist:
        pars = {parlist[n].upper(): parlist[n+1] for n in range(0, len(parlist), 2)}

    # This dict translates SarPy window names into a scipy window names.
    scipy_window = {'uniform': 'boxcar',
                    'hanning': 'hann',
                    }.get(window, window)

    scipy_pars = spectral_taper.Taper().default_pars.get(window.upper(), {}) if pars is None else pars

    for n in [n_evn, n_odd]:
        w_actual = spectral_taper.Taper(window, pars=pars, default_size=n).window_vals

        if scipy_window == 'boxcar':
            w_expect = sciwin.boxcar(n, sym=True)
        elif scipy_window == 'hamming':
            w_expect = sciwin.hamming(n, sym=True)
        elif scipy_window == 'hann':
            w_expect = sciwin.hann(n, sym=True)
        elif scipy_window == 'general_hamming':
            w_expect = sciwin.general_hamming(n, alpha=scipy_pars['ALPHA'], sym=True)
        elif scipy_window == 'kaiser':
            w_expect = sciwin.kaiser(n, beta=scipy_pars['BETA'], sym=True)
        elif scipy_window == 'taylor':
            w_expect = sciwin.taylor(n, nbar=scipy_pars['NBAR'], sll=np.abs(scipy_pars['SLL']), norm=True, sym=True)
        else:
            raise ValueError(f"Unknown window function {scipy_window}")

        assert np.allclose(w_actual, w_expect)


def test_make_sym_1d_window_exceptions():
    with pytest.raises(ValueError, match='Window type "SHAZAM" is not supported\\.'):
        spectral_taper.Taper("shazam")


def test_taper_get_vals():
    taper1 = spectral_taper.Taper("hamming", default_size=101)
    taper2 = spectral_taper.Taper("hamming", default_size=201)

    assert np.allclose(taper1.get_vals(201, sym=True), taper2.get_vals(201, sym=True))
    assert np.allclose(taper1.get_vals(201, sym=False), taper2.get_vals(201, sym=False))


@pytest.mark.parametrize("axis", ["Row", "Col"])
def test_get_sicd_wgt_funct(axis, mock_sicd_meta):
    nsamp = 51
    axis_mdata = {"Row": mock_sicd_meta.Grid.Row, "Col":  mock_sicd_meta.Grid.Col}[axis]

    alpha = axis_mdata.WgtType.Parameters["ALPHA"]
    w_expect = np.array(alpha + (1 - alpha) * np.cos(np.linspace(-np.pi, np.pi, nsamp)))
    w_actual = spectral_taper._get_sicd_wgt_funct(mock_sicd_meta, axis, desired_size=nsamp)
    assert np.allclose(w_actual, w_expect, atol=1.0e-3)

    axis_mdata.WgtType.WindowName = "UNIFORM"
    axis_mdata.WgtFunct = None
    w_expect = np.ones(nsamp)
    w_actual = spectral_taper._get_sicd_wgt_funct(mock_sicd_meta, axis, desired_size=nsamp)
    assert np.all(w_actual == w_expect)

    axis_mdata.WgtType.WindowName = 'SHAZAM'
    with pytest.raises(ValueError, match=f'SICD/Grid/{axis}/WgtFunct is not part of the SICD metadata'):
        spectral_taper._get_sicd_wgt_funct(mock_sicd_meta, axis, desired_size=nsamp)


@pytest.mark.parametrize("img_rows,img_cols,fft_sgn,skew_axis", [(128, 256, -1, 9), (129, 257, -1, 9),
                                                                 (128, 256, 1, 9), (129, 257, 1, 9),
                                                                 (128, 256, -1, 0), (128, 256, -1, 1)
                                                                 ])
def test_apply_spectral_taper(mock_sicd_meta, img_rows, img_cols, fft_sgn, skew_axis):
    class MockSICDReader():
        def __init__(self, mdata):
            self.sicd_meta = mdata

            nrows_support = int(mdata.Grid.Row.ImpRespBW * mdata.Grid.Row.SS * mdata.ImageData.NumRows)
            ncols_support = int(mdata.Grid.Col.ImpRespBW * mdata.Grid.Col.SS * mdata.ImageData.NumCols)

            row_dc_bin = mdata.ImageData.NumRows // 2
            row_edge_low = row_dc_bin - nrows_support // 2
            row_edge_hgh = row_dc_bin + nrows_support // 2

            col_dc_bin = mdata.ImageData.NumCols // 2
            col_edge_low = col_dc_bin - ncols_support // 2
            col_edge_hgh = col_dc_bin + ncols_support // 2

            cdata_fft = np.zeros(shape=(mdata.ImageData.NumRows, mdata.ImageData.NumCols), dtype=complex)
            cdata_fft[row_edge_low:row_edge_hgh+1,  col_edge_low:col_edge_hgh+1] = 1.0

            if mdata.Grid.Row.WgtFunct is not None:
                wgts = spectral_taper._fit_1d_window(mdata.Grid.Row.WgtFunct, row_edge_hgh - row_edge_low + 1)
                cdata_fft[row_edge_low:row_edge_hgh+1, :] *= wgts[:, np.newaxis]

            if mdata.Grid.Col.WgtFunct is not None:
                wgts = spectral_taper._fit_1d_window(mdata.Grid.Col.WgtFunct, col_edge_hgh - col_edge_low + 1)
                cdata_fft[:, col_edge_low:col_edge_hgh+1] *= wgts

            self.cdata = (scifft.fftshift(scifft.fft2(scifft.fftshift(cdata_fft)))
                          / (mdata.ImageData.NumRows * mdata.ImageData.NumCols))

            if mdata.Grid.Row.DeltaKCOAPoly is not None:
                self.apply_kspace_skew('Row', forward=True)

            if mdata.Grid.Row.DeltaKCOAPoly is not None:
                self.apply_kspace_skew('Col', forward=True)

        def apply_kspace_skew(self, axis, forward=True):
            axis_imdex = {'Row': 0, 'Col': 1}[axis]
            axis_mdata = {'Row': self.sicd_meta.Grid.Row, 'Col': self.sicd_meta.Grid.Col}[axis]

            xrow = ((np.arange(0, self.sicd_meta.ImageData.NumRows) - self.sicd_meta.ImageData.SCPPixel.Row)
                    * self.sicd_meta.Grid.Row.SS)
            ycol = ((np.arange(0, self.sicd_meta.ImageData.NumCols) - self.sicd_meta.ImageData.SCPPixel.Col)
                    * self.sicd_meta.Grid.Col.SS)

            delta_k_coa_poly = np.asarray(axis_mdata.DeltaKCOAPoly.Coefs)

            phs_delta_k_coa_poly = np.polynomial.polynomial.polyint(delta_k_coa_poly, axis=axis_imdex)
            phs_delta_k_vals = axis_mdata.Sgn * np.polynomial.polynomial.polygrid2d(xrow, ycol, phs_delta_k_coa_poly)

            arg = (-1 if forward else 1) * 1j * 2 * np.pi
            self.cdata *= np.exp(arg * phs_delta_k_vals)

        def __getitem__(self, item):
            return self.cdata

    mock_sicd_meta.Grid.Row.Sgn = fft_sgn
    mock_sicd_meta.Grid.Col.Sgn = fft_sgn

    mock_sicd_meta.Grid.Row.WgtType = None
    mock_sicd_meta.Grid.Row.WgtFunct = None
    mock_sicd_meta.Grid.Col.WgtType = None
    mock_sicd_meta.Grid.Col.WgtFunct = None

    delta_k_coa_poly = np.array([[0.3, -6.6e-04, 2.5e-6], [-3.2e-5, 1.3e-09, -3.5e-15]])

    if skew_axis == 0:
        mock_sicd_meta.Grid.Row.DeltaKCOAPoly = delta_k_coa_poly

    if skew_axis == 1:
        mock_sicd_meta.Grid.Col.DeltaKCOAPoly = delta_k_coa_poly

    mock_sicd_meta.ImageData.NumRows = img_rows
    mock_sicd_meta.ImageData.NumCols = img_cols
    mock_sicd_meta.ImageData.SCPPixel = (mock_sicd_meta.ImageData.NumRows//2, mock_sicd_meta.ImageData.NumCols//2)

    sicd_reader = MockSICDReader(mock_sicd_meta)

    taper_window = "Taylor"
    taper = spectral_taper.Taper(taper_window)

    cdata_in = copy.deepcopy(sicd_reader[:, :])
    max_in = np.max(np.abs(cdata_in))
    max_in_loc = np.where(np.abs(cdata_in) == max_in)
    max_in_row = max_in_loc[0][0]
    max_in_col = max_in_loc[1][0]

    # Make sure that the input data has complex conjugate symmetry
    r1 = 1 - cdata_in.shape[0] % 2
    c1 = 1 - cdata_in.shape[1] % 2
    assert max_in_row == cdata_in.shape[0] // 2 and max_in_col == cdata_in.shape[1] // 2
    if skew_axis != 0:
        assert np.allclose(cdata_in[r1:max_in_row, max_in_col], np.conj(cdata_in[-1:max_in_row:-1, max_in_col]))
    if skew_axis != 1:
        assert np.allclose(cdata_in[max_in_row, c1:max_in_col], np.conj(cdata_in[max_in_row, -1:max_in_col:-1]))

    cdata_out, mdata = spectral_taper.apply_spectral_taper(sicd_reader, taper_window)

    max_out = np.max(np.abs(cdata_out))
    max_out_loc = np.where(np.abs(cdata_out) == max_out)
    max_out_row = max_out_loc[0][0]
    max_out_col = max_out_loc[1][0]

    assert max_out_row == max_in_row and max_out_col == max_in_col

    # Make sure that the output side lobes are less than the input side lobes
    row_ipr_in = np.abs(cdata_in[max_out_row, :max_out_col-1] / cdata_in[max_out_row, max_out_col])
    col_ipr_in = np.abs(cdata_in[:max_out_row-1, max_out_col] / cdata_in[max_out_row, max_out_col])

    row_ipr_out = np.abs(cdata_out[max_out_row, :max_out_col-1] / cdata_out[max_out_row, max_out_col])
    col_ipr_out = np.abs(cdata_out[:max_out_row-1, max_out_col] / cdata_out[max_out_row, max_out_col])

    assert np.all(np.convolve(row_ipr_out, np.ones(3), mode='valid') <
                  np.convolve(row_ipr_in, np.ones(3), mode='valid'))
    assert np.all(np.convolve(col_ipr_out, np.ones(3), mode='valid') <
                  np.convolve(col_ipr_in, np.ones(3), mode='valid'))

    row_inv_osf = mdata.Grid.Row.ImpRespBW * mdata.Grid.Row.SS
    col_inv_osf = mdata.Grid.Col.ImpRespBW * mdata.Grid.Col.SS

    expected_max_in = row_inv_osf * col_inv_osf
    assert np.allclose(max_in, expected_max_in, rtol=0.001)

    expected_max_out = expected_max_in * np.mean(taper.window_vals)**2
    assert np.allclose(max_out, expected_max_out, rtol=0.02)

    expected_coh_pwr_gain = (expected_max_out / expected_max_in) ** 2
    assert np.allclose(mdata.Radiometric.RCSSFPoly[0][0] * expected_coh_pwr_gain, 1.0)

    rms_pwr_gain = np.mean(taper.window_vals**2)**2
    assert np.allclose(mdata.Radiometric.NoiseLevel.NoisePoly.Coefs[0][0] / rms_pwr_gain, 1.0)
    assert np.allclose(mdata.Radiometric.SigmaZeroSFPoly[0][0] * rms_pwr_gain, 1.0)
    assert np.allclose(mdata.Radiometric.BetaZeroSFPoly[0][0] * rms_pwr_gain, 1.0)
    assert np.allclose(mdata.Radiometric.GammaZeroSFPoly[0][0] * rms_pwr_gain, 1.0)


def test_apply_2d_spectral_taper(mock_sicd_meta, monkeypatch):
    def mock_apply_1d_spectral_taper(cdata, mdata, axis, window):
        cdata[axis] = window
        return cdata

    monkeypatch.setattr(spectral_taper, '_apply_1d_spectral_taper', mock_apply_1d_spectral_taper)

    row_scale = 2
    col_scale = 4

    mock_sicd_meta.Grid.Row.WgtFunct = np.full(shape=(11,), fill_value=row_scale)
    mock_sicd_meta.Grid.Col.WgtFunct = np.full(shape=(12,), fill_value=col_scale)
    window_vals = np.arange(13)

    cdata = {}
    cdata = spectral_taper._apply_2d_spectral_taper(cdata, mock_sicd_meta, window_vals)

    assert np.allclose(cdata['Row'], window_vals / row_scale)
    assert np.allclose(cdata['Col'], window_vals / col_scale)
