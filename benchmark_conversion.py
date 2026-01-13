#!/usr/bin/env python
"""
Benchmark script for SarPy conversion performance.

This script measures:
1. Micro-benchmarks for optimized components (polynomial fitting, XML parsing, etc.)
2. Full Sentinel-1 to SICD conversion performance

Usage:
    # Run micro-benchmarks only
    python benchmark_conversion.py --micro

    # Run full conversion benchmark
    python benchmark_conversion.py /path/to/sentinel1.zip

    # Run both
    python benchmark_conversion.py /path/to/sentinel1.zip --micro
"""

import os
import sys
import time
import tempfile
import tracemalloc
import hashlib
import argparse
import numpy as np
from contextlib import contextmanager
from typing import Dict, List


@contextmanager
def timer(name: str, results: Dict[str, List[float]]):
    """Context manager to time a block of code."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    if name not in results:
        results[name] = []
    results[name].append(elapsed)


def get_file_checksums(directory):
    """Get checksums of all files in directory for verification."""
    checksums = {}
    for fname in sorted(os.listdir(directory)):
        fpath = os.path.join(directory, fname)
        if os.path.isfile(fpath):
            with open(fpath, 'rb') as f:
                checksums[fname] = hashlib.md5(f.read()).hexdigest()
    return checksums


# =============================================================================
# MICRO-BENCHMARKS
# =============================================================================

def benchmark_polynomial_fitting(iterations: int = 100) -> Dict[str, float]:
    """Benchmark the two_dim_poly_fit function (optimized with polyvander2d)."""
    from sarpy.io.complex.utils import two_dim_poly_fit

    # Create test data similar to what's used in Sentinel-1 conversion
    # Typical grid is poly_order + 4 = 6 points in each dimension
    grid_size = 6
    x = np.linspace(-1000, 1000, grid_size)
    y = np.linspace(-500, 500, grid_size)
    x_2d, y_2d = np.meshgrid(x, y)
    x_flat = x_2d.flatten()
    y_flat = y_2d.flatten()

    # Simulate polynomial data with some noise
    z = 1.0 + 2.0*x_flat + 3.0*y_flat + 0.001*x_flat*y_flat + np.random.randn(x_flat.size) * 0.01

    results = {}

    # Warmup
    for _ in range(5):
        two_dim_poly_fit(x_flat, y_flat, z, x_order=2, y_order=2, x_scale=1e-3, y_scale=1e-3)

    # Benchmark
    for _ in range(iterations):
        with timer('two_dim_poly_fit', results):
            two_dim_poly_fit(x_flat, y_flat, z, x_order=2, y_order=2, x_scale=1e-3, y_scale=1e-3)

    return {
        'mean_ms': np.mean(results['two_dim_poly_fit']) * 1000,
        'std_ms': np.std(results['two_dim_poly_fit']) * 1000,
        'iterations': iterations,
        'note': 'Optimized with numpy.polynomial.polynomial.polyvander2d'
    }


def benchmark_xml_parsing(iterations: int = 100) -> Dict[str, float]:
    """Benchmark XML parsing (optimized to single-pass)."""
    from sarpy.io.xml.base import parse_xml_from_string

    # Sample SICD-like XML with namespaces
    xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
    <SICD xmlns="urn:SICD:1.3.0">
        <CollectionInfo>
            <Classification>UNCLASSIFIED</Classification>
            <ReleaseInfo>PUBLIC RELEASE</ReleaseInfo>
            <CollectorName>TestCollector</CollectorName>
            <CoreName>TestCore</CoreName>
            <CollectType>MONOSTATIC</CollectType>
            <RadarMode>
                <ModeType>SPOTLIGHT</ModeType>
            </RadarMode>
        </CollectionInfo>
        <ImageData>
            <PixelType>RE32F_IM32F</PixelType>
            <NumRows>1024</NumRows>
            <NumCols>2048</NumCols>
            <FirstRow>0</FirstRow>
            <FirstCol>0</FirstCol>
            <FullImage>
                <NumRows>1024</NumRows>
                <NumCols>2048</NumCols>
            </FullImage>
            <SCPPixel>
                <Row>512</Row>
                <Col>1024</Col>
            </SCPPixel>
        </ImageData>
        <GeoData>
            <EarthModel>WGS_84</EarthModel>
            <SCP>
                <LLH><Lat>0.0</Lat><Lon>0.0</Lon><HAE>0.0</HAE></LLH>
                <ECF><X>6378137.0</X><Y>0.0</Y><Z>0.0</Z></ECF>
            </SCP>
        </GeoData>
    </SICD>'''

    results = {}

    # Warmup
    for _ in range(5):
        parse_xml_from_string(xml_content)

    # Benchmark
    for _ in range(iterations):
        with timer('parse_xml_from_string', results):
            root, ns = parse_xml_from_string(xml_content)

    return {
        'mean_ms': np.mean(results['parse_xml_from_string']) * 1000,
        'std_ms': np.std(results['parse_xml_from_string']) * 1000,
        'iterations': iterations,
        'note': 'Optimized to single-pass parsing with namespace capture'
    }


def benchmark_format_conversion(iterations: int = 50) -> Dict[str, float]:
    """Benchmark IQ format conversion (optimized with slice-based indexing)."""
    from sarpy.io.general.format_function import ComplexFormatFunction

    # Create test data - typical chip size
    rows, cols = 256, 256
    data = np.random.randint(0, 32767, size=(rows, cols, 2), dtype=np.int16)

    fmt = ComplexFormatFunction(np.dtype('int16'), 'IQ', band_dimension=2)
    fmt._raw_shape = (rows, cols, 2)
    fmt._formatted_shape = (rows, cols)
    fmt._reverse_axes = ()
    fmt._transpose_axes = None

    subscript = (slice(None), slice(None), slice(None))

    results = {}

    # Warmup
    for _ in range(3):
        fmt._forward_functional_step(data, subscript)

    # Benchmark
    for _ in range(iterations):
        with timer('format_conversion', results):
            fmt._forward_functional_step(data, subscript)

    return {
        'mean_ms': np.mean(results['format_conversion']) * 1000,
        'std_ms': np.std(results['format_conversion']) * 1000,
        'iterations': iterations,
        'data_shape': f'{rows}x{cols}x2',
        'note': 'Optimized with slice-based indexing instead of numpy.take()'
    }


def benchmark_deep_copy(iterations: int = 50) -> Dict[str, float]:
    """Benchmark Serializable deep copy (optimized with direct deepcopy)."""
    from sarpy.io.complex.sicd_elements.SICD import SICDType
    from sarpy.io.complex.sicd_elements.CollectionInfo import CollectionInfoType, RadarModeType
    from sarpy.io.complex.sicd_elements.ImageData import ImageDataType
    from sarpy.io.complex.sicd_elements.GeoData import GeoDataType, SCPType
    from sarpy.io.complex.sicd_elements.blocks import RowColType, LatLonHAEType, XYZType

    # Create a minimal but representative SICD structure
    sicd = SICDType(
        CollectionInfo=CollectionInfoType(
            Classification='UNCLASSIFIED',
            CollectorName='TestCollector',
            CoreName='TestCore',
            CollectType='MONOSTATIC',
            RadarMode=RadarModeType(ModeType='SPOTLIGHT')
        ),
        ImageData=ImageDataType(
            PixelType='RE32F_IM32F',
            NumRows=1024,
            NumCols=2048,
            FirstRow=0,
            FirstCol=0,
            SCPPixel=RowColType(Row=512, Col=1024)
        ),
        GeoData=GeoDataType(
            EarthModel='WGS_84',
            SCP=SCPType(
                LLH=LatLonHAEType(Lat=0.0, Lon=0.0, HAE=0.0),
                ECF=XYZType(X=6378137.0, Y=0.0, Z=0.0)
            )
        )
    )

    results = {}

    # Warmup
    for _ in range(3):
        sicd.copy()

    # Benchmark
    for _ in range(iterations):
        with timer('deep_copy', results):
            sicd.copy()

    return {
        'mean_ms': np.mean(results['deep_copy']) * 1000,
        'std_ms': np.std(results['deep_copy']) * 1000,
        'iterations': iterations,
        'note': 'Optimized with direct copy.deepcopy() instead of to_dict/from_dict'
    }


def run_micro_benchmarks():
    """Run all micro-benchmarks."""
    print("\n" + "=" * 70)
    print("  MICRO-BENCHMARKS: Optimized Components")
    print("=" * 70)

    benchmarks = [
        ("Polynomial Fitting (two_dim_poly_fit)", benchmark_polynomial_fitting),
        ("XML Parsing (parse_xml_from_string)", benchmark_xml_parsing),
        ("IQ Format Conversion", benchmark_format_conversion),
        ("SICD Deep Copy", benchmark_deep_copy),
    ]

    results_summary = []

    for name, func in benchmarks:
        print(f"\n  Running: {name}...", end=" ", flush=True)
        try:
            results = func()
            print("OK")
            print(f"    Mean: {results['mean_ms']:.3f} ms (+/- {results['std_ms']:.3f} ms)")
            print(f"    Iterations: {results['iterations']}")
            if 'note' in results:
                print(f"    Note: {results['note']}")
            results_summary.append((name, results))
        except Exception as e:
            print(f"FAILED: {e}")
            results_summary.append((name, {'error': str(e)}))

    return results_summary


# =============================================================================
# FULL CONVERSION BENCHMARK
# =============================================================================

def benchmark_sentinel_reader(file_path: str, num_runs: int = 3) -> Dict:
    """Benchmark Sentinel-1 reader initialization (metadata parsing)."""
    from sarpy.io.complex.sentinel import SentinelReader

    results = {}

    print(f"\n  Benchmarking Sentinel-1 reader ({num_runs} runs)...")

    for run in range(num_runs):
        print(f"    Run {run + 1}/{num_runs}...", end=" ", flush=True)

        tracemalloc.start()
        with timer('reader_init', results):
            reader = SentinelReader(file_path)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        if run == 0:
            sicds = reader.get_sicds_as_tuple()
            num_sicds = len(sicds)
            # Get image dimensions
            total_pixels = sum(s.ImageData.NumRows * s.ImageData.NumCols for s in sicds)
            print(f"OK ({num_sicds} SICDs, {total_pixels/1e6:.1f}M pixels)")
        else:
            print("OK")

        reader.close()

    return {
        'mean_s': np.mean(results['reader_init']),
        'std_s': np.std(results['reader_init']),
        'num_runs': num_runs,
        'num_sicds': num_sicds,
        'total_pixels_millions': total_pixels / 1e6,
        'peak_memory_mb': peak / (1024 * 1024)
    }


def benchmark_full_conversion(input_file: str, output_dir: str = None) -> Dict:
    """Benchmark full conversion to SICD NITF files."""
    from sarpy.io.complex.converter import conversion_utility

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix='sarpy_benchmark_')
        cleanup = True
    else:
        cleanup = False

    print(f"\n  Running full conversion...")
    print(f"    Output: {output_dir}")

    tracemalloc.start()
    start_time = time.perf_counter()

    conversion_utility(input_file, output_dir)

    elapsed = time.perf_counter() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Get output info
    output_files = os.listdir(output_dir)
    total_output_size = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in output_files
    )

    result = {
        'elapsed_s': elapsed,
        'peak_memory_mb': peak / (1024 * 1024),
        'output_files': len(output_files),
        'total_output_mb': total_output_size / (1024 * 1024),
        'output_dir': output_dir
    }

    if cleanup:
        import shutil
        shutil.rmtree(output_dir)
        result['output_dir'] = '(cleaned up)'

    return result


def run_conversion_benchmark(input_file: str, num_reader_runs: int = 3):
    """Run full conversion benchmark."""
    print("\n" + "=" * 70)
    print("  CONVERSION BENCHMARK: Sentinel-1 to SICD")
    print("=" * 70)

    if not os.path.exists(input_file):
        print(f"\n  ERROR: File not found: {input_file}")
        return None

    file_size_mb = os.path.getsize(input_file) / (1024 * 1024)
    print(f"\n  Input file: {input_file}")
    print(f"  File size: {file_size_mb:.1f} MB")

    # Benchmark reader initialization
    reader_results = benchmark_sentinel_reader(input_file, num_reader_runs)
    print(f"\n  Reader initialization:")
    print(f"    Mean time: {reader_results['mean_s']:.2f} s (+/- {reader_results['std_s']:.2f} s)")
    print(f"    Peak memory: {reader_results['peak_memory_mb']:.1f} MB")
    print(f"    SICDs created: {reader_results['num_sicds']}")

    # Benchmark full conversion
    conversion_results = benchmark_full_conversion(input_file)
    print(f"\n  Full conversion:")
    print(f"    Time: {conversion_results['elapsed_s']:.2f} s")
    print(f"    Peak memory: {conversion_results['peak_memory_mb']:.1f} MB")
    print(f"    Output files: {conversion_results['output_files']}")
    print(f"    Total output: {conversion_results['total_output_mb']:.1f} MB")

    # Calculate throughput
    throughput_mpix_s = reader_results['total_pixels_millions'] / conversion_results['elapsed_s']
    print(f"\n  Throughput: {throughput_mpix_s:.2f} Mpixels/s")

    return {
        'reader': reader_results,
        'conversion': conversion_results,
        'throughput_mpix_s': throughput_mpix_s
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark SarPy performance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    # Run micro-benchmarks only
    python benchmark_conversion.py --micro

    # Benchmark Sentinel-1 conversion
    python benchmark_conversion.py /path/to/sentinel1.zip

    # Run both micro-benchmarks and conversion
    python benchmark_conversion.py /path/to/sentinel1.zip --micro

    # Multiple reader runs for more accurate timing
    python benchmark_conversion.py /path/to/sentinel1.zip --runs 5
        '''
    )
    parser.add_argument('file', nargs='?', help='Sentinel-1 file to benchmark')
    parser.add_argument('--micro', action='store_true', help='Run micro-benchmarks')
    parser.add_argument('--runs', type=int, default=3, help='Number of reader initialization runs')

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  SarPy Performance Benchmark")
    print("  Measuring optimizations from performance analysis")
    print("=" * 70)

    # Run micro-benchmarks
    if args.micro or not args.file:
        run_micro_benchmarks()

    # Run conversion benchmark if file provided
    if args.file:
        run_conversion_benchmark(args.file, num_reader_runs=args.runs)

    print("\n" + "=" * 70)
    print("  Benchmark Complete")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
