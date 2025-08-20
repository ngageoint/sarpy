#
# Copyright 2022 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
import collections
import unittest

import numpy as np

from sarpy.visualization import remap

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class NoOpRemap(remap.RemapFunction):
    def __init__(self):
        super().__init__(override_name="noop")

    def raw_call(self, data, **kwargs):
        return data


class TestRemap(unittest.TestCase):
    def setUp(self):
        remap._DEFAULTS_REGISTERED = False
        remap._REMAP_DICT.clear()

    def tearDown(self):
        remap._DEFAULTS_REGISTERED = False
        remap._REMAP_DICT.clear()

    def test_clip_cast(self):
        data = np.asarray([-100000, -128, -10, 0, 10, 127, 100000], dtype=np.float64)

        result = remap.clip_cast(data, np.int8)
        np.testing.assert_array_almost_equal(result, [-128, -128, -10, 0, 10, 127, 127])

        result = remap.clip_cast(data, np.int8, -100, 100)
        np.testing.assert_array_almost_equal(result, [-100, -100, -10, 0, 10, 100, 100])

        result = remap.clip_cast(data, np.int8, -500, 500)
        np.testing.assert_array_almost_equal(result, [-128, -128, -10, 0, 10, 127, 127])

        result = remap.clip_cast(data, np.int16)
        np.testing.assert_array_almost_equal(result, [-32768, -128, -10, 0, 10, 127, 32767])

    def test_amplitude_to_density_zeros(self):
        data = np.zeros(100, dtype=np.complex64)
        result = remap.amplitude_to_density(data.copy())
        np.testing.assert_array_equal(data, result)

    def test_amplitude_to_density(self):
        data = np.arange(100, dtype=np.complex64)
        result = remap.amplitude_to_density(data)
        self.assertTrue(np.all(np.isfinite(result)))
        with self.assertRaises(ValueError):
            remap.amplitude_to_density(data, dmin=-1)
        with self.assertRaises(ValueError):
            remap.amplitude_to_density(data, dmin=255)
        with self.assertRaises(ValueError):
            remap.amplitude_to_density(data, mmult=0)

    def test_RemapFunction(self):
        rf = remap.RemapFunction()
        self.assertEqual(rf.bit_depth, 8)
        self.assertEqual(rf.dimension, 0)
        self.assertEqual(rf.output_dtype, np.dtype(np.uint8))
        self.assertTrue(rf.are_global_parameters_set)
        self.assertTrue(isinstance(rf.name, str))
        self.assertGreater(len(rf.name), 0)
        with self.assertRaises(NotImplementedError):
            rf(np.arange(10))

        with self.assertRaises(NotImplementedError):
            rf.calculate_global_parameters_from_reader(None)

        rf = remap.RemapFunction(override_name="unit_test_remap")
        self.assertEqual(rf.name, "unit_test_remap")

        with self.assertRaises(ValueError):
            rf = remap.RemapFunction(override_name=123)

        with self.assertRaises(ValueError):
            rf = remap.RemapFunction(dimension=10)

        for bit_depth in [8.0, 16, 32]:
            rf = remap.RemapFunction(bit_depth=bit_depth)
            self.assertEqual(rf.output_dtype.kind, "u")
            self.assertEqual(rf.output_dtype.itemsize, bit_depth // 8)

        for bit_depth in [0, 15.0, 64]:
            with self.assertRaises(ValueError):
                remap.RemapFunction(bit_depth=bit_depth)

        # Check that casting and clipping occurs
        data = np.linspace(-1000, 1000)
        remapped = NoOpRemap()(data)
        self.assertEqual(remapped.dtype, np.dtype(np.uint8))
        np.testing.assert_array_equal(remapped, remap.clip_cast(data))

    def test_MonochromaticRemap(self):
        mr = remap.MonochromaticRemap()
        self.assertEqual(mr.bit_depth, 8)
        self.assertEqual(mr.max_output_value, 255)
        mr = remap.MonochromaticRemap(bit_depth=16)
        self.assertEqual(mr.max_output_value, (1 << 16) - 1)
        with self.assertRaises(ValueError):
            mr = remap.MonochromaticRemap(bit_depth=16, max_output_value=1 << 16)
        with self.assertRaises(ValueError):
            mr = remap.MonochromaticRemap(bit_depth=16, max_output_value=0)
        with self.assertRaises(NotImplementedError):
            mr.calculate_global_parameters_from_reader(None)
        with self.assertRaises(NotImplementedError):
            mr(np.random.uniform(100))

    def test_Density(self):
        with self.assertRaises(ValueError):
            remap.Density(dmin=-1)
        with self.assertRaises(ValueError):
            remap.Density(dmin=256)
        with self.assertRaises(ValueError):
            remap.Density(mmult=0.9)

        data = np.random.lognormal(size=1000).astype(np.complex128)
        dr = remap.Density()
        nominal = dr(data)
        self.assertEqual(nominal.dtype, np.uint8)
        double = dr(data * 2)
        np.testing.assert_array_equal(nominal, double)
        self.assertFalse(dr.are_global_parameters_set)
        self.assertTrue(remap.Density(data_mean=1).are_global_parameters_set)

    def test_Linear(self):
        data = np.linspace(2, 1, 512)
        lr = remap.Linear()
        self.assertFalse(lr.are_global_parameters_set)
        self.assertTrue(remap.Linear(min_value=1, max_value=2).are_global_parameters_set)
        nominal = lr(data)
        self.assertGreaterEqual(np.count_nonzero(nominal), data.size - 3)
        self.assertEqual(nominal.dtype, np.uint8)
        self.assertEqual(nominal.min(), 0)
        self.assertEqual(nominal.max(), 255)

        double = lr(data * 2)
        np.testing.assert_array_equal(nominal, double)

        lr.min_value = data.mean()
        with_min = lr(data)
        self.assertEqual(np.count_nonzero(with_min), data.size / 2 - 1)
        with self.assertRaises(ValueError):
            lr.min_value = np.inf
        lr.min_value = None
        self.assertGreaterEqual(np.count_nonzero(lr(data)), data.size - 3)

        lr.max_value = data.mean()
        with_max = lr(data)
        self.assertEqual(np.sum(with_max == 255), np.sum(data >= data.mean()))
        with self.assertRaises(ValueError):
            lr.max_value = np.inf
        lr.max_value = None
        np.testing.assert_array_equal(lr(data), nominal)
        double = lr(data * 2)
        np.testing.assert_array_equal(nominal, double)

    def test_Logarithmic(self):
        data = np.linspace(2, 1, 512)
        lr = remap.Logarithmic()
        self.assertFalse(lr.are_global_parameters_set)
        self.assertTrue(remap.Logarithmic(min_value=1, max_value=2).are_global_parameters_set)
        nominal = lr(data)
        self.assertEqual(nominal.dtype, np.uint8)
        self.assertEqual(nominal.min(), 0)
        self.assertEqual(nominal.max(), 255)

        lr.max_value = 1.5
        with_max = lr(data)
        self.assertEqual(nominal.min(), 0)
        np.testing.assert_array_equal(with_max[:256], 255)
        np.testing.assert_array_less(with_max[256:], 255)

        lr.max_value = None
        lr.min_value = 1.5
        with_max = lr(data)
        self.assertEqual(nominal.max(), 255)
        np.testing.assert_array_equal(with_max[256:], 0)

    def test_Logarithmic_const(self):
        data = np.full(1000, 1.0, dtype=np.complex64)
        lr = remap.Logarithmic()
        np.testing.assert_array_equal(lr(data), 0)

    def test_PEDF(self):
        self.assertFalse(remap.PEDF().are_global_parameters_set)
        self.assertTrue(remap.PEDF(data_mean=0.5).are_global_parameters_set)
        data = np.random.lognormal(size=1000).astype(np.complex128)
        pedf = remap.PEDF()(data)
        self.assertEqual(pedf.dtype, np.uint8)
        self.assertEqual(pedf.min(), 0)

    def test_NRL(self):
        self.assertFalse(remap.NRL().are_global_parameters_set)
        self.assertTrue(remap.NRL(stats=(0, 2, 1)).are_global_parameters_set)

        data = np.random.lognormal(size=1000).astype(np.complex128)
        nrl = remap.NRL()(data)
        self.assertEqual(nrl.dtype, np.uint8)
        self.assertEqual(nrl.min(), 0)
        self.assertEqual(nrl.max(), 255)

        with self.assertRaises(ValueError):
            remap.NRL(percentile=0)
        with self.assertRaises(ValueError):
            remap.NRL(percentile=100)
        with self.assertRaises(ValueError):
            remap.NRL(max_output_value=100, knee=101)

    def test_NRL_near_const(self):
        data = np.full(1000, 1.0, dtype=np.complex64)
        data[-1] += 1e-6
        with self.assertLogs('sarpy.visualization.remap', level='WARNING') as lc:
            nrl = remap.NRL()(data)
            self.assertTrue(any('at least significantly constant' in msg for msg in lc.output))
        expected = np.zeros(data.size, dtype=np.uint8)
        expected[-1] = 255
        np.testing.assert_array_equal(nrl, expected)

    def test_NRL_inf(self):
        data = np.full(1000, np.inf, dtype=np.complex64)
        nrl = remap.NRL()(data)
        np.testing.assert_array_equal(nrl, 0)

    def test_MonoRemaps(self):
        data = np.random.lognormal(size=1000).astype(np.complex128)
        adata = np.abs(data)
        nominal = remap.Density()(data)
        brighter = remap.Brighter()(data)
        darker = remap.Darker()(data)
        self.assertEqual(nominal.dtype, np.uint8)
        self.assertEqual(brighter.dtype, np.uint8)
        self.assertEqual(darker.dtype, np.uint8)
        self.assertTrue(np.all(brighter >= nominal))
        self.assertGreater(np.mean(brighter), np.mean(nominal))
        self.assertTrue(np.all(darker <= nominal))
        self.assertLess(np.mean(darker), np.mean(nominal))

        hc = remap.High_Contrast()(data)
        self.assertEqual(hc.dtype, np.uint8)
        self.assertGreater(np.sum(hc == 0), np.sum(nominal == 0))
        self.assertGreater(np.sum(hc == 255), np.sum(nominal == 255))

        linear = remap.Linear()(data)
        self.assertEqual(linear.dtype, np.uint8)
        self.assertAlmostEqual(max(linear / adata), 255 / max(adata))

        log = remap.Logarithmic()(data)
        self.assertEqual(log.dtype, np.uint8)
        self.assertTrue(np.all(linear <= log))

        nrl = remap.NRL()(data)
        self.assertTrue(np.all(nrl >= linear))

    def test_LUTRemap(self):
        data = np.concatenate((np.arange(256), np.arange(256)[::-1]))
        lut = np.zeros((256, 3), dtype=np.uint8)
        lut[:, 0] = np.arange(256)
        lut[:, 1] = np.arange(256)[::-1]
        lut[:, 2] = np.roll(np.arange(256), 128)

        lutr = remap.LUT8bit(mono_remap=remap.Linear(), lookup_table=lut)
        result = lutr(data)
        np.testing.assert_array_equal(result.shape, data.shape + (3,))
        np.testing.assert_array_equal(result[:256], lut)
        np.testing.assert_array_equal(result[256:], lut[::-1])

        class NonMono(remap.RemapFunction):
            pass

        with self.assertRaises(ValueError):
            remap.LUT8bit(mono_remap=NonMono, lookup_table=lut)

    @unittest.skipIf(not MATPLOTLIB_AVAILABLE, "matplotlib not available")
    def test_LUTRemap_matplotlib(self):
        data = np.arange(0, 512, 2)[::-1]

        lutr = remap.LUT8bit(mono_remap=remap.Linear(), lookup_table="binary")
        result = lutr(data)
        np.testing.assert_array_equal(result.shape, data.shape + (3,))
        np.testing.assert_array_equal(result[:, 0], result[:, 1])
        np.testing.assert_array_equal(result[:, 0], result[:, 2])
        np.testing.assert_array_equal(result[:, 0], result[:, 2])
        self.assertTrue(np.all(np.abs(result[:, 0] - np.arange(256)) <= 1))

    def test_remap_names(self):
        default_names = remap.get_remap_names()
        self.assertIn("nrl", default_names)
        self.assertIn("density", default_names)
        self.assertIn("high_contrast", default_names)
        self.assertIn("brighter", default_names)
        self.assertIn("darker", default_names)
        self.assertIn("linear", default_names)
        self.assertIn("log", default_names)
        self.assertIn("pedf", default_names)
        self.assertIn("nrl_16", default_names)
        self.assertIn("density_16", default_names)
        self.assertIn("high_contrast_16", default_names)
        self.assertIn("brighter_16", default_names)
        self.assertIn("darker_16", default_names)
        self.assertIn("linear_16", default_names)
        self.assertIn("log_16", default_names)
        self.assertIn("pedf_16", default_names)

        if MATPLOTLIB_AVAILABLE:
            self.assertIn("viridis", default_names)
            self.assertIn("magma", default_names)
            self.assertIn("rainbow", default_names)
            self.assertIn("bone", default_names)

        noop_remap = NoOpRemap()
        remap.register_remap(noop_remap)
        updated_names = remap.get_remap_names()
        self.assertEqual(set(updated_names) - set(default_names), {"noop"})
        with self.assertRaises(TypeError):
            remap.register_remap(lambda x: x)

        another_noop_remap = NoOpRemap()
        remap.register_remap(another_noop_remap)
        self.assertIs(noop_remap, remap.get_registered_remap("noop"))  # didn't overwrite
        remap.register_remap(another_noop_remap, overwrite=True)
        self.assertIs(another_noop_remap, remap.get_registered_remap("noop"))  # did overwrite

    def test_get_registered_remap(self):
        with self.assertRaises(KeyError):
            remap.get_registered_remap("__fake__")
        self.assertEqual(remap.get_registered_remap("__fake__", "default", 8 ), "default")

    def test_get_registered_remap_required_param_only(self):
        self.assertEqual(remap.get_registered_remap("linear" ).name, "linear")
     
    def test_get_registered_remap_required_param_default(self):
        self.assertEqual(remap.get_registered_remap("linear", "default" ).name, "linear")
 
    def test_get_registered_remap_required_param_default_bit_depth(self):
        self.assertEqual(remap.get_registered_remap("linear" ).name, "linear")
        self.assertEqual(remap.get_registered_remap("linear" ).bit_depth, 8)


    def test_get_registered_remap_bitdepth_param(self):
        self.assertEqual(remap.get_registered_remap("linear", bit_depth= 16 ).name, "linear")
        self.assertEqual(remap.get_registered_remap("linear", bit_depth= 16 ).bit_depth, 16)
 
    def test_get_registered_remap_falure_bitdepth_param(self):
        with self.assertRaises(KeyError):
            remap.get_registered_remap("linear", bit_depth= 32 )

    def test_get_registered_remap_falure_not_registered(self):
        with self.assertRaises(KeyError):
            remap.get_registered_remap("steve" )
    
    def test_get_remap_list(self):
        remap_list = remap.get_remap_list()
        self.assertSetEqual(set(item[0] for item in remap_list), set(remap.get_remap_names()))
        self.assertIsInstance(dict(remap_list)["density"], remap.Density)

    def test_flat_interface(self):
        data = np.random.lognormal(size=1000).astype(np.complex128)
        with self.assertWarns(DeprecationWarning):
            np.testing.assert_array_equal(remap.density(data), remap.Density()(data))
        with self.assertWarns(DeprecationWarning):
            np.testing.assert_array_equal(remap.brighter(data), remap.Brighter()(data))
        with self.assertWarns(DeprecationWarning):
            np.testing.assert_array_equal(remap.darker(data), remap.Darker()(data))
        with self.assertWarns(DeprecationWarning):
            np.testing.assert_array_equal(remap.high_contrast(data), remap.High_Contrast()(data))
        with self.assertWarns(DeprecationWarning):
            np.testing.assert_array_equal(remap.linear(data), remap.Linear()(data))
        with self.assertWarns(DeprecationWarning):
            np.testing.assert_array_equal(remap.log(data), remap.Logarithmic()(data))
        with self.assertWarns(DeprecationWarning):
            np.testing.assert_array_equal(remap.pedf(data), remap.PEDF()(data))
        with self.assertWarns(DeprecationWarning):
            np.testing.assert_array_equal(remap.nrl(data), remap.NRL()(data))
