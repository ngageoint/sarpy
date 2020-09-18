"""
Script for repairing any potential broken SICD DES header - should work for any
environment with sarpy installed
"""

import argparse
from sarpy.io.complex.sicd import SICDDetails

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


def repair(in_file):
    try:
        details = SICDDetails(in_file)
    except Exception as e:
        print('Treating file {} as a SICD failed with error {}'.format(in_file, e))
        return

    try:
        stat = details.repair_des_header()
    except Exception as e:
        print('Repairing DES header effort for file {} failed with error {}'.format(in_file, e))
        return

    if stat == 0:
        print(
            'UNCERTAIN: NITF file {} is not apparently a SICD, or DES Subheader '
            'is of an unexpected format for evaluation.'.format(in_file))
    elif stat == 1:
        print('SUCCESS: No change was required for file {}'.format(in_file))
    elif stat == 2:
        print('SUCCESS: DES subheader information was successfully changed for file {}'.format(in_file))
    elif stat == 3:
        print('FAILURE: DES subheader information was NOT successfully changed for file {}'.format(in_file))
    else:
        print('Got unknown status {} for file {}'.format(stat, in_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Repair SICD DES Header information (if necessary).")
    parser.add_argument('files', metavar='filename', nargs='+', help="a SICD file")
    args = parser.parse_args()

    for fil in args.files:
        repair(fil)
