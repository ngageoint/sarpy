import contextlib
import copy
import pathlib
 
import numpy as np
import numpy.testing
import pytest
 
import sarpy.consistency.cphd_consistency
import sarpy.io.phase_history.converter
from sarpy.io.phase_history.cphd import CPHDReader, CPHDReader0_3, CPHDWriter1
 
import tests
 
 
CPHD_FILE_TYPES = tests.find_test_data_files(pathlib.Path(__file__).parent / 'cphd_file_types.json')
for path in CPHD_FILE_TYPES.get('CPHD', []):
    print( "joz: {}".format( path ))
    if pathlib.Path(path).name == 'dynamic_stripmap_ci2.cphd':
        CI2_CPHD = path
        break
else:
    CI2_CPHD = None
 
 
@pytest.mark.parametrize('cphd_path', CPHD_FILE_TYPES.get('CPHD',[]))
def test_open_phase_history( cphd_path ):
 
    print( cphd_path )
    reader = sarpy.io.phase_history.converter.open_phase_history( cphd_path )
    assert isinstance(reader, CPHDReader)
    assert reader.reader_type == 'CPHD'
