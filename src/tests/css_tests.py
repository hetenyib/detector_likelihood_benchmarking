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
from src.layout_templates import parallelogram, diamond
from src.CSS_direct import Hex_Code, create_CSS_circuit as direct_css
from src.simulation import error_rate_sim

@pytest.mark.parametrize('sub_rounds',
                         [24,25,26,27,28,29])
@pytest.mark.parametrize('basis', ['primary', 'dual'])
def test_parallelogram_commuting(sub_rounds, basis):
    '''
    Remember, planar css needs pauli_cycle 2 and kesselring bdr.
    '''
    layout = parallelogram(2, 2, pauli_cycle=2, basis=basis)
    circuit = direct_css(layout, sub_rounds)
    
    try:
        circuit.detector_error_model()
    except ValueError:
        raise AttributeError(f'{sub_rounds} rounds CSS code did not commute.')
    else:
        assert True    
# End test_commuting


@pytest.mark.parametrize('sub_rounds',
                         [24,25,26,27,28,29])
@pytest.mark.parametrize('basis', ['primary', 'dual'])
def test_diamond_commuting(sub_rounds, basis):
    '''
    Remember, planar css needs pauli_cycle 2 and kesselring bdr.
    '''
    layout = diamond(2, basis=basis)
    circuit = direct_css(layout, sub_rounds)
    
    try:
        circuit.detector_error_model()
    except ValueError:
        raise AttributeError(f'{sub_rounds} rounds CSS code did not commute.')
    else:
        assert True    
# End test_commuting


@pytest.mark.parametrize('basis', ['primary', 'dual'])
@pytest.mark.parametrize('border', ['direct'])
def test_parallelogram_threshold(basis, border):
    '''
    Runs two error rate simulations to make sure error suppressed with distance.
    '''
    sub_rounds = 18
    shots = int(1e4)

    small_layout = parallelogram(2, 2, pauli_cycle=2, basis=basis, border=border)
    small_circuit = Hex_Code(small_layout, sub_rounds)
    large_layout = parallelogram(8, 8, pauli_cycle=2, basis=basis, border=border)
    large_circuit = Hex_Code(large_layout, sub_rounds)

    small_error_rate = error_rate_sim(small_circuit, shots)
    large_error_rate = error_rate_sim(large_circuit, shots)

    assert small_error_rate / large_error_rate > 6
# End test_threshold


@pytest.mark.parametrize('basis', ['primary', 'dual'])
def test_diamond_threshold(basis):
    '''
    Runs two error rate simulations to make sure error suppressed with distance.
    This ensures a threshold is being reached above 1e-3 (true for implemented code).
    '''
    sub_rounds = 18
    shots = int(1e4)

    small_layout = diamond(2, basis=basis)
    small_circuit = Hex_Code(small_layout, sub_rounds)
    large_layout = diamond(8,basis=basis)
    large_circuit = Hex_Code(large_layout, sub_rounds)

    small_error_rate = error_rate_sim(small_circuit, shots)
    large_error_rate = error_rate_sim(large_circuit, shots)

    assert small_error_rate / large_error_rate > 6
# End test_threshold



@pytest.mark.parametrize('basis', ['primary', 'dual'])
@pytest.mark.parametrize('size', range(2, 6))
def test_diamond_noiseless_detectors(basis, size):
    '''
    Runs two error rate simulations to make sure error suppressed with distance.
    '''
    sub_rounds = 18
    shots = 1024

    layout = diamond(size, basis=basis, noise=[0,0,0,0,0])
    code = Hex_Code(layout, sub_rounds)

    syns, log_ops = code.sample(shots)

    for s in syns:
        assert sum(s) == 0, 'A non-zero noiseless detector was found.'
    assert sum(log_ops) == 0, 'Logical Operator noiselessly flipped.'
# End test_threshold
