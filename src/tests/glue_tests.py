# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
from src.qiskit_glue import *
from src.simulation import run_aer_sim
from src.CSS_direct import Hex_Code
import src.layout_templates as templates
import pytest



@pytest.mark.parametrize('size', [2,3,4,5])
@pytest.mark.parametrize('sub_rounds', range(6,12))
def test_get_detector_zeros(size, sub_rounds):
    '''
    Testing to make sure that all dtrack results in zero syndromes with all zero check operators
    '''
    layout = templates.diamond(size)
    hex_code = Hex_Code(layout, sub_rounds)
    cross_code = Cross_Platform_Code(hex_code, 433, 12)

    fake_mmts = get_memory_needs(hex_code.circuit)[0] * '0'
    syns = cross_code.dtrack.get_detector_values(fake_mmts)

    assert sum(syns) == 0, f'Summing zeros yielded {sum(syns)} in get_detector_values for size {size} and {sub_rounds} sub_rounds.'
# End test_get_detector_zeros


@pytest.mark.parametrize('size', [2]) # Size 3 too big for simulator
@pytest.mark.parametrize('rounds', range(18, 25))
@pytest.mark.parametrize('corner', [4, 8, 20, 24])
def test_noiseless_detectors_eagle_diamond(size, rounds, corner):
    '''
    Takes the test stim circuit, runs a qiskit noiseless sim,
    then checks to make sure all detectors are 0.
    This test is only for two locations on a eagle device.
    'corner' defines qiskit index for the mapping.
    '''
    # Get stim circuit
    layout = templates.diamond(size)
    stim_code = Hex_Code(layout, sub_rounds=rounds)

    # Set up and run qiskit sim
    code = Cross_Platform_Code(stim_code, 
                                device_size=127, 
                                qiskit_index=corner)
    counts = run_aer_sim(code.qiskit_circuit, shots=1024)

    # Check detectors
    for count in counts:
        detector_values = code.dtrack.get_detector_values(count)
        log_op = code.dtrack.get_log_op(count)
        assert sum(detector_values) == 0, f'A detector was detected for {count}'
        assert log_op == 0, 'Logical op was 1 in a noiseless case.'
# End test_noiseless_detectors


def test_half_init_plaqs():
    '''
    Saving test after by hand finding values of detector_targets during debugging.
    '''
    device_size = 127
    qiskit_index = 4
    size = 3
    sub_rounds = 6

    layout = templates.diamond(size, basis='primary')
    hex_code = Hex_Code(layout, sub_rounds=sub_rounds)
    code = Cross_Platform_Code(hex_code, 
                           device_size=device_size, 
                           qiskit_index=qiskit_index, 
                           stim_index=min(layout.data_qubits),
                           generic_reset=False,
                           reflect=True)
    
    assert sorted(code.dtrack.detector_targets[15]) == [32, 34, 35]
    assert sorted(code.dtrack.detector_targets[16]) == [36, 37, 38]
    assert sorted(code.dtrack.detector_targets[17]) == [39, 40, 41]
    assert sorted(code.dtrack.detector_targets[18]) == [31, 45]
# End test_half_init_plaqs
