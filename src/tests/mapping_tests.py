# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import pytest
from src.device_mapping import *


#@pytest.mark.parametrize('register_index, stim_index, qiskit_index',
#                         [()])
def test_Mapping_getters():
    '''
    index%10 == 1 for stim, 2 for register, and 3 for qiskit indexes.
    tens place is index for a coordinate added.
    '''
    # Creating mapping object
    map = Mapping(127)
    for t in range(10):
        map.add_qubit(t*10 + 1, t*10 + 2, t*10 + 3)
    
    for t in range(10):
        assert map.get_stim_index(register_index=10*t + 2) == 10*t + 1
        assert map.get_stim_index(qiskit_index=10*t + 3) == 10*t + 1
        assert map.get_register_index(stim_index=10*t + 1) == 10*t + 2
        assert map.get_register_index(qiskit_index=10*t + 3) == 10*t + 2
        assert map.get_qiskit_index(register_index=10*t + 2) == 10*t + 3
        assert map.get_qiskit_index(stim_index=10*t + 1) == 10*t + 3
# End test_Mapping_getters
