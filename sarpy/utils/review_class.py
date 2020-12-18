import sys
import functools
from collections import defaultdict
import os
import pkgutil
from importlib import import_module


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


print_func = print


def traverse_module_classification(package_name, results_dict):
    module = import_module(package_name)
    path, fil = os.path.split(module.__file__)
    if fil.startswith('__init__.py'):
        # iterate over module children
        for sub_module in pkgutil.walk_packages([path, ]):
            _, sub_module_name, _ = sub_module
            sub_name = "{}.{}".format(package_name, sub_module_name)
            traverse_module_classification(sub_name, results_dict)
    else:
        # this is a "leaf" module, so inspect for classification definition
        class_str = getattr(module, '__classification__', '__NO_CLASSIFICATION__')
        results_dict[class_str].append(package_name)


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

    def argparse_formatter_factory(prog):
        return argparse.ArgumentDefaultsHelpFormatter(prog, width=100)

    parser = argparse.ArgumentParser(
        description='Utility to create a report for displaying package __classification__ values',
        formatter_class=argparse_formatter_factory)
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
