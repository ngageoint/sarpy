from sarpy import sarpy_support_dir
from sarpy.utils import file_utils

geoid_2008_1_fname = file_utils.get_path_from_subdirs(sarpy_support_dir, ['geoid_models',
                                                                          'egm2008-1',
                                                                          'geoids',
                                                                          'egm2008-1.pgm'])
