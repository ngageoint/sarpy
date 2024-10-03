import json
import os
import logging

parent_path = os.environ.get('SARPY_TEST_PATH', None)
if parent_path == 'NONE':
    parent_path = None
if parent_path is not None:
    parent_path = os.path.expanduser(parent_path)

if parent_path is not None and not os.path.isdir(parent_path):
    raise IOError('SARPY_TEST_PATH is given as {}, but is not a directory'.format(parent_path))


def find_test_data_files(test_json_file):
    """
    Find the test data files listed in the specified test JSON file.

    Parameters
    ----------
    test_json_file : str | pathlib.Path
        The full path specification to a JSON file that specifies unit test data files.

    Returns
    -------
    dict:
        A dictionary of files specified in the JSON file, if they exist in
        the path specified by the environmental variable SARPY_TEST_PATH.
    """
    test_data_files = {}
    with open(test_json_file, 'r') as fi:
        the_files = json.load(fi)
        for the_type in the_files:
            valid_entries = []
            for entry in the_files[the_type]:
                the_file = parse_file_entry(entry)
                if the_file is not None:
                    valid_entries.append(the_file)
            test_data_files[the_type] = valid_entries
    return test_data_files


def parse_file_entry(entry, default='absolute'):
    """
    Evaluate input as a path for file used in a unit test.

    Parameters
    ----------
    entry : None|dict
        dict of the form {'path': <value>', 'path_type': 'relative' or 'absolute'}
    default : str
        The default value for 'path_type', if it is not provided.

    Returns
    -------
    None|str
        The absolute path if the evaluated path exists, or `None` if not.
    """

    if entry is None:
        return None

    if not isinstance(entry, dict):
        raise ValueError('Got unexpected input.')
    if 'path' not in entry:
        raise KeyError('Input must have key "entry"')
    path_type = entry.get('path_type', default).lower()

    if path_type == 'absolute':
        the_file = os.path.expanduser(entry['path'])
    elif path_type == 'relative':
        if parent_path is None:
            logging.warning('Environment variable SARPY_TEST_PATH unset, but relative path identified in unit test')
            the_file = None
        else:
            the_file = os.path.join(parent_path, entry['path'])
    else:
        raise ValueError('value associated with "path_type" must be one of "absolute" or "relative"')

    if the_file is None:
        return None
    elif os.path.exists(the_file):
        return the_file
    else:
        return None
