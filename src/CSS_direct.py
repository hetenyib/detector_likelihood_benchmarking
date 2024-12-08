# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import stim 
import pymatching as pm
import numpy as np

from src.layout_planar_main import Planar_Layout
from src.layout_direct_borders import get_side_truncated_links
from src.measurement_tracker import *
from src.CSS_detectors import *
from src.logical_operator import *


class Hex_Code():
    '''
    CSS implementation from the Kesselring paper.
    '''
    def __init__(self, 
                 layout:Planar_Layout, 
                 sub_rounds, 
                 init_noise=True, 
                 ro_noise=True, 
                 ss_noise=True):
        '''
        
        '''
        #assert layout.border == 'kesselring', 'Only kesselring border implemented.'
        self.layout = layout
        self.sub_rounds = int(sub_rounds)
        
        self.circuit = create_CSS_circuit(layout, sub_rounds, 
                                          init_noise=init_noise, 
                                          ss_noise=ss_noise, 
                                          ro_noise=ro_noise)
        #print(self.circuit)
        self.sampler = self.circuit.compile_detector_sampler()
        self.dem = self.circuit.detector_error_model()
        self.match = pm.Matching(self.dem)
    # End __init__


    def sample(self, shots):
        '''
        easier sampler call.
        '''
        return self.sampler.sample(shots, separate_observables=True)
    # End sample

### End H3 ###


def create_CSS_circuit(layout:Planar_Layout, 
                      sub_rounds, 
                      init_noise=True, 
                      ro_noise=True, 
                      ss_noise=True) -> stim.Circuit:
    '''
    Creates full circuit, broken into main parts through other methods,
    main parts being initialization, steady state, and readout.
    '''
    circuit = stim.Circuit()
    mtrack = measurement_tracker()

    assert sub_rounds >= 4, 'Choose at least 4 subrounds.'

    # Initialization
    special_4 = True if sub_rounds == 4 else False
    circuit += generate_qubit_coords(layout)
    circuit += initialize(layout, mtrack, noise=init_noise, special_4=special_4)

    # Steady State
    full_cycles = max((sub_rounds // 6 - 2), 0)
    if full_cycles > 0:
        circuit += max((sub_rounds // 6 - 2), 0) * full_round(layout, mtrack, noise=ss_noise)

    # Readout
    if sub_rounds >= 12:
        last_r = sub_rounds % 6 - 1 + 6
        for r in range(sub_rounds % 6 + 6):
            last_round = last_r == r
            circuit += sub_round(r, layout, mtrack, noise=ro_noise, last_round=last_round)
    elif sub_rounds == 4:
        pass # Special case to push low number of sub_rounds, taken care of in initialization.
    else:
        #for r in range(sub_rounds % 6):
        last_r = sub_rounds % 6 - 1
        for r in range(sub_rounds % 6):
            last_round = last_r == r
            circuit += sub_round(r, layout, mtrack, noise=ro_noise, last_round=last_round)
    circuit += readout(layout, mtrack, sub_rounds % 6, noise=ro_noise)

    return circuit
# End create_CSS_circuit


def readout(layout:Planar_Layout, mtrack:measurement_tracker, sub_round, noise=True):
    '''
    Readsout correct logical observable in a given round.
    '''
    ro = stim.Circuit()

    # Adjust frame (if not already in Z)
    if layout.basis == 'dual':
        ro.append('H', layout.data_qubits)
        if noise:
            for qubit in layout.data_qubits:
                ro.append('DEPOLARIZE1', qubit, layout.noise_dict[qubit]['gate'])
            #ro.append('DEPOLARIZE1', layout.data_qubits, layout.single_gate_noise)
    if noise:
        for qubit in layout.data_qubits:
            ro.append('X_ERROR', qubit, layout.noise_dict[qubit]['RO'])
        #ro.append('X_ERROR', layout.data_qubits, layout.readout_noise)

    # Readout
    ro.append('M', layout.data_qubits)
    mtrack.add_measurement(layout.data_qubits, 'Readout')

    # Add last detectors
    ro += FTRO_detectors(layout, mtrack, sub_round)

    # Add logical observable
    logical_observable_indexes = [layout.observable_data_qubits[l]
                                  for l in range(len(layout.observable_data_qubits))
                                  if get_logical_op(sub_round, 
                                                    code_type='CSS', 
                                                    basis=layout.basis)[l%6] != '_']
    
    log_obs_targets = mtrack.get_mmt_targets(logical_observable_indexes, -1)
    ro.append('OBSERVABLE_INCLUDE', 
                        [stim.target_rec(target)
                         for target in log_obs_targets], 0)
    
    return ro
# End readout


def initialize(layout:Planar_Layout, 
               mtrack:measurement_tracker, 
               noise=True,
               special_4=False):
    '''
    Resetting data qubits, initializing basis, and running first full logical
    operator cycle to get the system into a steady state of subrounds.
    '''
    init = stim.Circuit()

    # Reset all data qubits
    init.append('R', layout.data_qubits + [index for index in layout.link_dict])
    if noise:
        for qubit in layout.data_qubits:
            init.append('X_ERROR', qubit, layout.noise_dict[qubit]['init'])
        #init.append('X_ERROR', layout.data_qubits, layout.init_noise)
    init.append('TICK')

    # Initialize logical op
    if layout.basis == 'dual':
        init.append_operation('H', layout.data_qubits)
        if noise:
            for qubit in layout.data_qubits:
                init.append('DEPOLARIZE1', qubit, layout.noise_dict[qubit]['gate'])
        #init.append_operation('DEPOLARIZE1', layout.data_qubits, layout.single_gate_noise)
        #init.append_operation('TICK')
    
    # Then Go through full round adjusting detectors as necessary.
    if layout.basis == 'primary':
        init += sub_round(-1, layout, mtrack, noise=noise, include_detectors=False, include_observable=False)
        init += get_init_check_detectors(layout, mtrack)
        init += sub_round(0, layout, mtrack, noise=noise, include_detectors=False)
        init += sub_round(1, layout, mtrack, noise=noise, include_detectors=True, partial_detector=True)
        init += sub_round(2, layout, mtrack, noise=noise, include_detectors=False)
        init += sub_round(3, layout, mtrack, noise=noise, include_detectors=True, last_round=special_4)

    elif layout.basis == 'dual':
        init += sub_round(0, layout, mtrack, noise=noise, include_detectors=False, include_observable=False)
        init += get_init_check_detectors(layout, mtrack)
        init += sub_round(1, layout, mtrack, noise=noise, include_detectors=False)
        init += sub_round(2, layout, mtrack, noise=noise, include_detectors=True, partial_detector=True)
        init += sub_round(3, layout, mtrack, noise=noise, include_detectors=False, last_round=special_4)

    if not special_4:
        init += sub_round(4, layout, mtrack, noise=noise, include_detectors=True)
        init += sub_round(5, layout, mtrack, noise=noise, include_detectors=True)

    return init
# End initialize


def full_round(layout, 
               mtrack, 
               noise=True, 
               include_detectors=True, 
               include_observable=True) -> stim.Circuit:
    '''
    Produces a full round of 6 sub-rounds for easy calling.
    '''
    full = stim.Circuit()
    for round in range(6):
        full += sub_round(round, 
                          layout, 
                          mtrack,
                          noise=noise, 
                          include_detectors=include_detectors, 
                          include_observable=include_observable)
    return full
# End full_round


def sub_round(sub_round:int, 
              layout:Planar_Layout, 
              mtrack:measurement_tracker,
              noise=True, 
              include_observable=True, 
              include_detectors=True,
              partial_detector=False,
              last_round=False) -> stim.Circuit:
    '''
    Takes link type as argument (0=Red, 1=Blue, 2=Green)
    puts together string the does one subround of the H3 code
    '''
    sub_round_circuit = stim.Circuit()

    # Measure link operators
    sub_round_circuit += measure_links(layout, mtrack, sub_round, noise=noise, last_round=last_round)

    # Adjust logical observable by adding in links
    if ((include_observable and sub_round % 2 == 1 and layout.basis == 'primary')
     or (include_observable and sub_round % 2 == 0 and layout.basis == 'dual')):
        sub_round_circuit.append("OBSERVABLE_INCLUDE", 
                                [stim.target_rec(m) 
                                 for m in mtrack.get_mmt_targets(layout.observable_links, -1)], 0)

    # Detectors & Shift coords
    if include_detectors:
        sub_round_circuit += get_plaquette_detectors(sub_round, layout, mtrack, partial=partial_detector)

    sub_round_circuit.append("SHIFT_COORDS", [], [0, 0, 1])

    return sub_round_circuit
# End sub_round


def measure_links(layout:Planar_Layout, 
                  mtrack:measurement_tracker, 
                  sub_round:int,
                  noise=True,
                  last_round=False) -> stim.Circuit:
    '''
    Takes a list of links to measure
    Returns circuit squence to measure all of the link operators of given type
    '''
    link_mmt = stim.Circuit()
    link_type = sub_round % 3
    x_frame = True if sub_round % 2 == 0 else False
    link_indexes = [index for index in layout.link_dict 
                    if layout.link_dict[index].color_type == link_type]
    # If X frame, y extreme bdrys included, if Z frame, x extreme bdrys
    truncated_indexes = [index for index in get_side_truncated_links(layout, x_frame)
                         if layout.truncated_links[index].color_type == link_type]

    # Setting Correct Frame, X frame needed for even rounds.
    frame_adjust = stim.Circuit()
    if x_frame:
        frame_adjust.append('H', layout.data_qubits)
        if noise: 
            for qubit in layout.data_qubits:
                frame_adjust.append('DEPOLARIZE1', qubit, layout.noise_dict[qubit]['gate'])
            #frame_adjust.append('DEPOLARIZE1', layout.data_qubits, layout.single_gate_noise)
    link_mmt += frame_adjust

    # CNOTS 
    cnot_pairs_0, other_0, cnot_pairs_1, other_1 = [],[],[],[]
    for l in link_indexes:
        link = layout.link_dict[l]
        #cnot_pairs_0.append((link.adj0, link.index))
        cnot_pairs_0.append(link.adj0)
        cnot_pairs_0.append(link.index)
        other_0.append(link.adj1)
        #cnot_pairs_1.append((link.adj1, link.index))
        cnot_pairs_1.append(link.adj1)
        cnot_pairs_1.append(link.index)
        other_1.append(link.adj0)

    # First direction
    link_mmt.append('CNOT', cnot_pairs_0)
    if noise:
        for p in range(len(cnot_pairs_0) // 2):
            pair = (cnot_pairs_0[2 * p], cnot_pairs_0[2 * p + 1])
            if pair[1] in layout.noise_dict[pair[0]]['2-gate']:
                two_gate_noise = layout.noise_dict[pair[0]]['2-gate'][pair[1]]
            else:
                two_gate_noise = layout.noise_dict[pair[0]]['2-gate']['default']
            link_mmt.append('DEPOLARIZE2', pair, two_gate_noise)
        #link_mmt.append('DEPOLARIZE2', cnot_pairs_0, layout.two_gate_noise)
    
    # Second direction
    link_mmt.append('CNOT', cnot_pairs_1)
    if noise:
        for p in range(len(cnot_pairs_1) // 2):
            pair = (cnot_pairs_1[2 * p], cnot_pairs_1[2 * p + 1])
            if pair[1] in layout.noise_dict[pair[0]]['2-gate']:
                two_gate_noise = layout.noise_dict[pair[0]]['2-gate'][pair[1]]
            else:
                two_gate_noise = layout.noise_dict[pair[0]]['2-gate']['default']
            link_mmt.append('DEPOLARIZE2', pair, two_gate_noise)
        #link_mmt.append('DEPOLARIZE2', cnot_pairs_1, layout.two_gate_noise)

    # Measurements
    link_mmt.append('TICK')
    if noise:
        for link in link_indexes + truncated_indexes:
            link_mmt.append('X_ERROR', link, layout.noise_dict[link]['RO'])
        #link_mmt.append('X_ERROR', link_indexes + truncated_indexes, layout.readout_noise)
    link_mmt.append('M', link_indexes + truncated_indexes)
    mtrack.add_measurement(link_indexes + truncated_indexes)
    if noise:
        for qubit in layout.data_qubits:
            link_mmt.append('DEPOLARIZE1', qubit, layout.noise_dict[qubit]['idle'])
        #link_mmt.append('DEPOLARIZE1', layout.data_qubits, layout.idle_noise)

    # Reset aux qubits for next round
    if not last_round:
        next_round_link_indexes = [index for index in layout.link_dict 
                        if layout.link_dict[index].color_type == (link_type + 1) % 3]
        link_mmt.append('R', next_round_link_indexes)
        if noise: 
            for link in next_round_link_indexes:
                link_mmt.append('X_ERROR', link, layout.noise_dict[link]['RO'])
            #link_mmt.append('X_ERROR', next_round_link_indexes, layout.init_noise)
        link_mmt.append('TICK')

    # Returning Frame   
    link_mmt += frame_adjust

    return link_mmt
# End measure_links


def generate_qubit_coords(layout:Planar_Layout):
    '''
    Uses link definitions to find all locations of data & ancilla qubits
    then defines their coordinates.
    Need to check link coord, and both adjacents (which need to avoid double counting)
    '''
    all_qubits = {}
    for l in layout.link_dict:
        link = layout.link_dict[l]
        x, y = layout.coord_from_index(link.index)
        x0, y0 = layout.coord_from_index(link.adj0)
        x1, y1 = layout.coord_from_index(link.adj1)
        if link.index not in all_qubits:
            all_qubits[link.index] = (x, y)
        if link.adj0 not in all_qubits:
            all_qubits[link.adj0] = (x0, y0)
        if link.adj1 not in all_qubits:
            all_qubits[link.adj1] = (x1, y1)

    #for l in layout.truncated_links:
    #    all_qubits[l] = layout.coord_from_index(l)
    
    coords = stim.Circuit()
    for index in all_qubits:
        x, y = all_qubits[index][0], all_qubits[index][1]
        coords.append("QUBIT_COORDS", index, (x, y))
    return coords
# End generate_qubit_coords
