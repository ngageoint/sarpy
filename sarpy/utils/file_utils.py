import os
from functools import reduce
from typing import Union
import glob


def make_dir_if_not_exists(dir_to_create    # type: str
                           ):               # type: (...) -> None
    if not os.path.exists(dir_to_create):
        os.makedirs(dir_to_create)


def get_all_folders_in_dir(directory        # type: str
                           ):               # type: (...) -> list
    return list(filter(os.path.isdir, [os.path.join(directory, item) for item in os.listdir(directory)]))


def get_all_files_in_dir(directory,         # type: str
                         extensions=None    # type: Union[str, list]
                         ):
    sanitized_extensions = []
    if type(extensions) == (type([])):
        for extension in extensions:
            sanitized_extensions.append(extension.replace(".", "").replace("*", ""))
    elif type(extensions) == (type("")):
        sanitized_extensions.append(extensions.replace(".", "").replace("*", ""))
    elif extensions is None:
        sanitized_extensions.append("*")
    else:
        raise TypeError("extensions should either be a string or list of strings")

    files = []
    for extension in sanitized_extensions:
        fnames = glob.glob(directory + os.path.sep + "*." + extension)
        files = files + fnames
    return files


def get_path_from_subdirs(base_dir,     # type: str
                          subdirs       # type: list
                          ):            # type: (...) -> str
    paths_list = [base_dir]
    paths_list.extend(subdirs)
    abs_path = reduce(os.path.join, paths_list)
    return abs_path
