from __future__ import print_function
import sys
import functools
from collections import defaultdict

import pkgutil
from importlib import import_module

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


print_func = print


def traverse_module_classification(package_name, results_dict):
    def evaluate(the_module, the_name):
        class_str = getattr(the_module, '__classification__', '__NO_CLASSIFICATION__')
        results_dict[class_str].append(the_name)

    module = import_module(package_name)

    if hasattr(module, '__path__'):
        for details in pkgutil.walk_packages(module.__path__, package_name+'.'):
            _, module_name, is_pkg = details
            if is_pkg:
                # don't evaluate the presence of class string for packages
                continue

            # noinspection PyBroadException
            sub_module = import_module(module_name)
            evaluate(sub_module, module_name)
    else:
        evaluate(module, package_name)


def check_classification(parent_package, results_dict=None):
    if results_dict is None:
        results_dict = defaultdict(list)
    traverse_module_classification(parent_package, results_dict)
    return results_dict


def log_package_classification(parent_package, dest=sys.stdout):
    global print_func
    print_func = functools.partial(print, file=dest)

    results_dict = check_classification(parent_package)
    for class_str in sorted(results_dict.keys()):
        print_func(class_str)
        for entry in results_dict[class_str]:
            print_func('\t', entry)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Utility to create a report for displaying package __classification__ values',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-p', '--package', default='sarpy',
                        help="package or module name, should be a subpackage of sarpy")
    parser.add_argument('-o', '--output', default='stdout',
                        help="'stdout', 'string', or an output file")
    args = parser.parse_args()

    if args.output == 'stdout':
        # Send output to stdout
        log_package_classification(args.package, dest=sys.stdout)
    else:
        # Send output to file
        with open(args.output, 'w') as f:
            log_package_classification(args.package, dest=f)
