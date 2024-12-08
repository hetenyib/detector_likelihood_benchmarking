# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import qiskit
import stim
import pymatching as pm
import numpy as np
from typing import Tuple
from src.device_mapping import *
from src.CSS_direct import Hex_Code
from qiskit.circuit.library import XGate, RZGate
from qiskit.transpiler import PassManager, InstructionDurations
from qiskit.transpiler.passes import DynamicalDecoupling, PadDynamicalDecoupling, ALAPScheduleAnalysis
from qiskit import transpile
import src.layout_templates as templates
from datetime import datetime 
import pickle


class Cross_Platform_Code():

    def __init__(self, 
                 code:Hex_Code, 
                 device_size:int, 
                 qiskit_index, 
                 stim_index=None,
                 reflect=True,
                 generic_reset=True):
        '''
        Initialization takes a stim circuit and layout
        '''
        self.stim_circuit = code.circuit
        self.sub_rounds = code.sub_rounds
        self.layout = code.layout
        self.dem = code.dem
        self.match = code.match # Probably have some sort of feedback for this later.
        self.location = qiskit_index

        self.dtrack = detector_tracker()
        if stim_index is None:
            stim_index = min(self.layout.data_qubits)
        self.map = get_qiskit_mapping(self.stim_circuit, 
                                      code.layout, 
                                      stim_index, 
                                      qiskit_index, 
                                      reflect, 
                                      device_size=device_size)

        # Determining size and creating shell
        num_qubits = len(self.stim_circuit.get_final_qubit_coordinates())
        total_mmts, needed_c_ifs = get_memory_needs(self.stim_circuit)
        num_cl_registers = total_mmts + needed_c_ifs
        self.qiskit_circuit = qiskit.QuantumCircuit(num_qubits, num_cl_registers)

        # Converting to qiskit Circuit
        mmt_count = 0
        for instruction in self.stim_circuit:
            if type(instruction) == stim._stim_sse2.CircuitInstruction:
                mmt_count = add_gate(self, 
                                    instruction, 
                                    mmt_count,
                                    total_mmts,
                                    generic_reset=generic_reset)
            elif type(instruction) == stim._stim_sse2.CircuitRepeatBlock:
                for _ in range(instruction.repeat_count):
                    for repeat_instruction in instruction.body_copy():
                        mmt_count = add_gate(self, 
                                            repeat_instruction, 
                                            mmt_count,
                                            total_mmts,
                                            generic_reset=generic_reset)
    # End __init__ (for now)

    def sample(self, shots):
        '''
        easier sampler call.
        '''
        return self.stim_circuit.compile_detector_sampler().sample(shots, separate_observables=True)
    # End sample


### End Cross_Platform_Code ###


def add_calibration_noise(cross_code:Cross_Platform_Code, size:int, backend, date=None):
    '''
    Given a cross_code and a backend, returns a new cross_code
    with noise model based on calibration data for specific backend
    '''
    if type(backend) == str:
        f = open("noise_dictionaries.obj", "rb")
        dicts = pickle.load(f)
        f.close()
        if backend in dicts:
            noise_dict = dicts[backend][size][cross_code.location]
        else:
            raise KeyError("That is not a backend stored in the file. (sherbrooke or torino)")
    else:
        if date is None:
            noise_dict = get_noise_dict_from_backend(backend, cross_code.map)
        else:
            noise_dict = get_noise_dict_from_backend(backend, cross_code.map, date)

    if cross_code.layout.template == 'diamond':
        layout = templates.diamond(size, 
                                   noise_dict=noise_dict, 
                                   basis=cross_code.layout.basis)
    elif cross_code.layout.template == 'parallelogram':
        layout = templates.parallelogram(size, size,
                                         noise_dict=noise_dict, 
                                         basis=cross_code.layout.basis)
    elif cross_code.layout.template == 'basicbox':
        if cross_code.style == 'weather':
            layout = templates.basic_box(size//2, size, noise_dict=noise_dict)
        elif cross_code.style == 'test':
            layout = templates.basic_box(1, 1, noise_dict=noise_dict)
        else:
            layout = templates.basic_box(size, size, noise_dict=noise_dict)

    hex_code = Hex_Code(layout, 
                        cross_code.sub_rounds)
    cross_code = Cross_Platform_Code(hex_code, 
                                     cross_code.map.device_size, 
                                     cross_code.location)
    return cross_code
# End add_calibration_noise


class detector_tracker():
    '''
    Needed to keep track of detectors after stim circuit converted to qiskit.
    This is so that when a result is gained from qiskit, this object
    can be used to determine detector values from link and data qubit mmts.
    '''
    def __init__(self):
        self.detector_targets = []
        self.logical_observable = []
        #self.detector_coords = []
        #self.t = 0
    # End __init__

    def __str__(self):
        msg = f'<detector_tracker with {len(self.detector_targets)} detectors'
        msg += 'with a logical observable>' if len(self.logical_observable) > 0 else '>'
        return msg
    # End __str__

    def num_detectors(self):
        return len(self.detector_targets)
    # End num_detectors

    def add_detector(self, targets):
        '''
        Needs to be passed classical register values for targets.
        Stores the classical register values to look up later.
        '''
        self.detector_targets.append(targets)
    # End add_detector

    def include_observable(self, targets):
        '''
        Need to keep track of what to include in the logical observable.
        '''
        self.logical_observable += targets
    # End add_detector

    def get_log_op(self, bit_str):
        '''
        Returns the value of the logical observable measured by the circuit.
        '''
        # Need to reverse the order because the backend flips order.
        bit_list = [int(bit) for bit in bit_str[::-1]]
        included = [bit_list[t] for t in self.logical_observable]
        log_op = sum(included)

        return log_op % 2
    # End get_log_op


    def get_detector_values(self, bit_str):
        '''
        Takes bitstring of a single shot from qiskit output 
        and returns array of detector values, 
        sized to throw straight into pymatching.
        The detector_targets stored already have stored indexes of 
        location in the bit string
        '''
        # Need to reverse the order because the backend flips order.
        bit_list = [int(bit) for bit in bit_str[::-1]]
        detector_values = []
        for targets in self.detector_targets:
            mmts_to_include = [bit_list[t] for t in targets]
            detector_values.append(sum(mmts_to_include) % 2)
        return detector_values
    # End get_detectors

### End detector_tracker class ###

def basic_stim_to_qiskit(stim_circuit:stim.Circuit) -> Tuple[qiskit.QuantumCircuit, detector_tracker]:
    '''
    Is this used? Was this just for testing and now active code lives in object???
    '''
    print('HELLOOOOOOOO.')
    # Determining size and creating shell
    num_qubits = len(stim_circuit.get_final_qubit_coordinates())

    # TODO:Get number of classical mmts needed
    total_mmts = 21
    needed_c_ifs = 12
    num_cl_registers = total_mmts + needed_c_ifs
    qiskit_circuit = qiskit.QuantumCircuit(num_qubits, num_cl_registers)
    dtrack = detector_tracker()
    # Converting to qiskit Circuit
    mmt_count = 0
    for instruction in stim_circuit:
        if type(instruction) == stim._stim_sse2.CircuitInstruction:
            mmt_count = add_gate(qiskit_circuit, 
                                instruction, 
                                dtrack,
                                mmt_count,
                                total_mmts)
        elif type(instruction) == stim._stim_sse2.CircuitRepeatBlock:
            print('REPEAT BLOCK', instruction, 'Hello.')
            for _ in range(instruction.repeat_count):
                for instruction in instruction.body_copy():
                    mmt_count = add_gate(qiskit_circuit, 
                                        instruction, 
                                        dtrack,
                                        mmt_count,
                                        total_mmts)

    return qiskit_circuit, dtrack
# End basic_stim_to_qiskit


def add_gate(code:Cross_Platform_Code, 
             instruction:stim._stim_sse2.CircuitInstruction,
             mmt_count:int,
             total_mmts:int,
             generic_reset=False) -> int:
    '''
    
    '''
    circuit = code.qiskit_circuit
    dtrack = code.dtrack
    map = code.map
    gate = instruction.name
    stim_targets = [t.value for t in instruction.targets_copy()]
    # When targets need to point to a register
    if gate in ['R', 'H', 'M', 'CX']:
        targets = [map.get_register_index(stim_index) 
                   for stim_index in stim_targets]
    # Targets pointing backward for observable or detector
    else:
        targets = stim_targets
    # Need to convert to register targets
    # Included so far:
    # ['R', 'H', 'M', 'CX', 'TICK', 'DETECTOR', 'OBSERVABLE_INCLUDE']
    if gate == 'R':
        if generic_reset:
            circuit.reset(targets)
        else:
            # Msr
            #print('R', end=' ')
            for t in range(len(targets)):
                circuit.measure(targets[t], total_mmts + t)
                #print(targets[t], total_mmts + t, end=', ')
            circuit.barrier(range(circuit.num_qubits))
            #print('.')
            # Reset
            for t in range(len(targets)):
                circuit.x(targets[t]).c_if(total_mmts + t, 1)
    elif gate == 'H':
        circuit.h(targets)
    elif gate == 'CX':
        for t in range(len(targets) // 2):
            circuit.cx(targets[2 * t], targets[2 * t + 1])
    elif gate == 'M':
        registers = [m for m in range(mmt_count, mmt_count + len(targets))]
        circuit.measure(targets, registers)
        #print('M ->', [(code.map.get_stim_index(register_index=targets[t]), 
        #               registers[t]) 
        #               for t in range(len(targets))])
        mmt_count += len(targets)
    elif gate == 'TICK':
        circuit.barrier(range(circuit.num_qubits))
    elif gate == 'DETECTOR':
        dtrack.add_detector([mmt_count + t for t in targets])
        #print('DETECTOR',[mmt_count + t for t in targets], mmt_count, targets)
    elif gate == 'OBSERVABLE_INCLUDE':
        dtrack.include_observable([mmt_count + t for t in targets])
    #elif gate == 'SHIFT_COORDS':
    #    dtrack.t += 1
    #else:
    #    print('OTHER',gate)
    return mmt_count
# End add_gate


def get_memory_needs(stim_circuit:stim.Circuit):
    '''
    Goes through a stim circuit and defines:
    - total number of measurements
    - Max resets at a given time.
    Both quantities needed for initialization of qiskit circuit.
    '''
    total_mmts = 0
    max_resets = 0
    for instruction in stim_circuit:
        if type(instruction) == stim._stim_sse2.CircuitInstruction:
            if instruction.name == 'M':
                total_mmts += len(instruction.targets_copy())
            if instruction.name == 'R':
                num_resets = len(instruction.targets_copy())
                if max_resets < num_resets:
                    max_resets = num_resets
        elif type(instruction) == stim._stim_sse2.CircuitRepeatBlock:
            for _ in range(instruction.repeat_count):
                for repeat_instruction in instruction.body_copy():
                    if repeat_instruction.name == 'M':
                        total_mmts += len(repeat_instruction.targets_copy())
                    if repeat_instruction.name == 'R':
                        num_resets = len(repeat_instruction.targets_copy())
                        if max_resets < num_resets:
                            max_resets = num_resets
    return total_mmts, max_resets
# End get_memory_needs


def add_dynamic_decoupling(transpiled_circuit, 
                           cross_code, 
                           backend, 
                           style='XX',
                           staggered=True):
    '''
    Adding a dynamic decoupling pass of the flavor of 
    two X gates possitioned to cancel out long phase noise.
    Needs a second pass and retranspile to make scheduling 
    work with delays that are multiples of 16 timesteps.
    '''
    # Dynamic Decoupling pass
    durations = InstructionDurations().from_backend(backend)
    if style == 'XX':
        dd_sequence = [XGate(), XGate()]
    elif style == 'XXXX':
        dd_sequence = [XGate(), XGate(), XGate(), XGate()]
    elif style == 'XZX':
        dd_sequence = [XGate(), RZGate(np.pi), XGate(), XGate(), RZGate(np.pi), XGate()]
    else:
        raise KeyError(f'"{style}" is not an implemented dd_sequence.')

    if staggered:
        pmanager = PassManager([ALAPScheduleAnalysis(durations),
                                PadDynamicalDecoupling(durations, 
                                                dd_sequence, 
                                                qubits=[cross_code.map.get_qiskit_index(stim_index=si)
                                                        for si in cross_code.layout.data_qubits])])
    else:
        pmanager = PassManager([DynamicalDecoupling(durations, 
                                                dd_sequence, 
                                                qubits=[cross_code.map.get_qiskit_index(stim_index=si)
                                                        for si in cross_code.layout.data_qubits])])

    dd_circuit = pmanager.run(transpiled_circuit)

    # Making sure delays are multiple of 16
    total_delay = [{q: 0 for q in dd_circuit.qubits} for _ in range(2)]
    for gate in dd_circuit.data:
        if gate[0].name == "delay":
            q = gate[1][0]
            t = gate[0].params[0]
            total_delay[0][q] += t
            new_t = 16 * np.ceil((total_delay[0][q] - total_delay[1][q]) / 16)
            total_delay[1][q] += new_t
            gate[0].params[0] = new_t

    # Re-transpiling
    dd_circuit = transpile(dd_circuit, 
                            backend, 
                            scheduling_method='alap')


    return dd_circuit
# End add_dynamic_decoupling



def get_noise_dict_from_backend(backend, map:Mapping, date=None):
    '''
    Takes a given backend and a mapping object and returns a noise dictionary
    to pass to a new Cross_Platform_Code/Hex_Code to have accurate backend 
    specific noise information based on calibration data.
    '''
    # Pre-reqs
    noise_dict = {}
    round_time = 2000e-9 
    if date is None:
        t = None
    else:
        t = datetime(day=int(date[-2:]), month=int(date[-4:-2]), year=int(date[:4]))
    properties = backend.properties(datetime=t)

    # Initializing each qubit with single qubit noise
    for qubit in map.qubits:
        # Defining ROI error
        ROI_error = properties.readout_error(qubit['qiskit_index'])

        # Defining idle error
        t1 = properties.t1(qubit['qiskit_index'])
        t2 = properties.t2(qubit['qiskit_index'])
        t_time = min(t1, t2)
        idle_error = 1 - np.exp(-round_time / t_time)

        # Updating dictionary
        noise_dict[qubit['stim_index']] = {}
        noise_dict[qubit['stim_index']]['init'] = ROI_error
        noise_dict[qubit['stim_index']]['idle'] = idle_error
        noise_dict[qubit['stim_index']]['RO'] = ROI_error
        noise_dict[qubit['stim_index']]['gate'] = 1e-4
        noise_dict[qubit['stim_index']]['2-gate'] = {'default': 1e-2}
        
    # Two qubit gate noise
    all_qiskit_indexes = [qubit['qiskit_index'] for qubit in map.qubits]
    for pair in backend.coupling_map:
        # Making sure connection is in circuit
        if pair[0] in all_qiskit_indexes and pair[1] in all_qiskit_indexes:
            # Getting two qubit gate error
            if backend.name == 'ibm_torino':
                two_gate_error = properties.gate_error('cz', pair)
            else:
                two_gate_error = properties.gate_error('ecr', pair)

            if two_gate_error > .5:
                two_gate_error = .5
            # Updating dictionary 
            stim_index_0 = map.get_stim_index(qiskit_index=pair[0])
            stim_index_1 = map.get_stim_index(qiskit_index=pair[1])
            noise_dict[stim_index_0]['2-gate'][stim_index_1] = two_gate_error
            noise_dict[stim_index_1]['2-gate'][stim_index_0] = two_gate_error

    return noise_dict
# End get_noise_dict_from_backend
