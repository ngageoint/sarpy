from collections import defaultdict
import os
import pkgutil
from importlib import import_module


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


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


def log_package_classification(parent_package, out_file):
    results_dict = check_classification(parent_package)
    with open(out_file, 'w') as fi:
        for class_str in sorted(results_dict.keys()):
            fi.write('{}\n'.format(class_str))
            for entry in results_dict[class_str]:
                fi.write('\t{}\n'.format(entry))


if __name__ == '__main__':
    log_package_classification('sarpy', os.path.expanduser('~/Desktop/sarpy_class_test.txt'))
