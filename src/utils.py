from qiskit import transpile
from qiskit_aer.noise.errors.quantum_error import QuantumChannelInstruction
from qiskit_aer.noise import depolarizing_error, pauli_error
from qiskit.circuit.library import XGate, RZGate
from qiskit.transpiler import PassManager, InstructionDurations
from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDynamicalDecoupling
from qiskit import QuantumCircuit
from typing import  List

from stim import Circuit as StimCircuit
from stim import target_rec as StimTarget_rec

import src.surface_code_decoder as surface_code_decoder

import numpy as np

from math import comb
import scipy.optimize as optimize
from matplotlib import cm, colors
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def LogFail_of_d_p(code_circuit_class,
                   theta: float, phi: float,  logical: str,
                   dist_list: List[int], error_list: List[float],
                   max_shots: int, shot_batch: int, max_num_fail: int,
                   T_over_d: float = 1, max_fail_rate: float = 0.49, DL = True,printing=False, spatial_boundaries=False, **kwargs):
    Log_fail_d_p = []
    for d in dist_list:
        Log_fail_p =[]
        for error in error_list:
            if len(Log_fail_p) == 0 or Log_fail_p[-1][1] < max_fail_rate:                
                Log_fail_p.append([error] + Log_fail_error_model(code_circuit_class=code_circuit_class, d=d, T=int(T_over_d*d), logical=logical,
                                            singleQerror = error*np.cos(theta)*np.cos(phi), CXerror = error*np.cos(theta)*np.cos(phi),
                                            idleerror = error*np.cos(theta)*np.sin(phi), 
                                            Rerror = error*np.sin(theta), 
                                            max_shots = max_shots, max_num_fail = max_num_fail, shot_batch = shot_batch,
                                            spatial_boundaries=spatial_boundaries,
                                            DL=DL, LF_error = True, **kwargs))
            if printing:
                print(d,Log_fail_p[-1])

        Log_fail_d_p.append([d,Log_fail_p])
    
    return theta,phi,Log_fail_d_p

def Log_fail_error_model(code_circuit_class, d, T, logical, singleQerror, CXerror, idleerror, Rerror, max_shots, max_num_fail, shot_batch, DL, LF_error, noise_dict = None, spatial_boundaries=False, **kwargs):
    if noise_dict:
        code = code_circuit_class(d=d, T=T, logical_observable=logical,
                                    singleQerror = 0, CXerror = 0, idleerror = 0, Rerror = 0,
                                    **kwargs)
        noisy_circuit = noisify_circuit(code.circuit, noise_dict=noise_dict)
        if 'build_stim_circuit' in dir(code):
            code.circuit = noisy_circuit
            stim_circuit = code.build_stim_circuit()
        else:
            stim_circuit = get_stim_circuits(noisy_circuit)[0][0]
            for detector in code.error_sensitive_events:
                stim_circuit.append("DETECTOR",[StimTarget_rec(measind-((d**2-1)*T+d**2)) for measind in detector])
            stim_circuit.append("OBSERVABLE_INCLUDE", [StimTarget_rec(code.measuredict[(pos,T)]-((d**2-1)*T+d**2)) for pos in code.edge_qubit_pos],0)

    else:
        code = code_circuit_class(d=d, T=T, logical_observable=logical,
                                    singleQerror = singleQerror, CXerror = CXerror, idleerror = idleerror, Rerror = Rerror,
                                    **kwargs)
        stim_circuit = code.stim_circuit
    
    stim_DEM = stim_circuit.detector_error_model(decompose_errors=True,approximate_disjoint_errors=True,ignore_decomposition_failures=True)
    matching = surface_code_decoder.detector_error_model_to_pymatching_graph(stim_DEM)
    num_fail = 0
    num_shots = 0
    det_likelihood = 0
    while num_shots<max_shots and num_fail<max_num_fail:
        num_shots += shot_batch
        detector_samples = stim_circuit.compile_detector_sampler().sample(shot_batch, append_observables=True)

        actual_obs_batch = detector_samples[:,-1:]
        predicted_flip_batch = matching.decode_batch(detector_samples) #logical is included as a last detector, but is given for the boundary node anyway
        correction_batch = (predicted_flip_batch != actual_obs_batch)
        num_fail+=sum(correction_batch)[0]
        if DL:
            det_likelihood += code.detector_likelihood(detector_samples=detector_samples,spatial_boundaries=spatial_boundaries,temporal_boundaries=False) #np.mean(sum(detector_samples)/shot_batch)
    
    result_list = [num_fail/num_shots]
    if LF_error:
        result_list += [np.sqrt(num_fail)/num_shots]
    if DL:
        result_list += [det_likelihood*shot_batch/num_shots]
    return result_list

def Threshold_from_LogFail(Log_fail_d_p, error_list_cutoff: float = 0.495, DL = True):
    # determine the largest error-list index of the shortest error list
    max_eind = len(Log_fail_d_p[0][1])
    for dind in range(len(Log_fail_d_p)):
        error_list, pLlist = np.transpose(Log_fail_d_p[dind][1])[:2]
        eind = 0
        while pLlist[eind]<error_list_cutoff and eind<len(error_list)-1:
            eind+=1
        if eind<max_eind:
            max_eind=eind

    error_list = [Log_fail_d_p[-1][1][i][0] for i in range(max_eind)] # redefine error_list as the shortest one
    dist_list = [Log_fail_d_p[dind][0] for dind in range(len(Log_fail_d_p))]
    
    rates_below_threshold = []
    rates_above_threshold = []
    pL_range = []
    for errorind in range(len(error_list)):
        pLofdist = [Log_fail_d_p[dind][1][errorind][1] for dind in range(len(dist_list))]
        pL_range.append(max(pLofdist) - min(pLofdist))
        if min(pLofdist) > 0.: # if the logical error rate is still 0 for some distance, we must be far from threshold
            if pLofdist[::-1] == sorted(pLofdist):
                rates_below_threshold.append(error_list[errorind]) # we only need the max
            if pLofdist == sorted(pLofdist):
                rates_above_threshold.append(error_list[errorind]) # we only need the min
    if rates_below_threshold == [] or rates_above_threshold == []:
        result = [0]*len(Log_fail_d_p[0][1][0])
    else:
        lower_bound_ind = error_list.index(max(rates_below_threshold))
        upper_bound_ind = error_list.index(min(rates_above_threshold))
        pL_range_lower = pL_range[lower_bound_ind]
        pL_range_upper = pL_range[upper_bound_ind]
        p_threshold = (pL_range_upper*max(rates_below_threshold)+pL_range_lower*min(rates_above_threshold))/(pL_range_lower + pL_range_upper)
        p_threshold_error = max(p_threshold-max(rates_below_threshold),min(rates_above_threshold)-p_threshold)
        if DL:
            DLmin = np.transpose(Log_fail_d_p[0][1])[-1][lower_bound_ind] #lowest distance, lower bound
            DLmax = np.transpose(Log_fail_d_p[-1][1])[-1][upper_bound_ind] #highest distance, upper bound
            result = p_threshold, p_threshold_error, (DLmax+DLmin)/2
        else:
            result = p_threshold, p_threshold_error

    return result

def noisify_circuit(circuits, noise_dict: dict):
    """
    Inserts error operations into a circuit according to a pauli noise model.
    Handles idling errors in the form of custom gates "idle_#" which are assumed to
    encode the identity gate only.
    qc = QuantumCircuit(1, name='idle_1')
    qc.i(0)
    idle_1 = qc.to_instruction()

    Args:
        circuits: Circuit or list thereof to which noise is added.
        noise_dict: dictionary of error probabilities provided by get_noise_dict_from_backend().

    Returns:
        noisy_circuits: Corresponding circuit or list thereof.
    """

    single_circuit = isinstance(circuits, QuantumCircuit)
    if single_circuit:
        circuits = [circuits]

    
    noisy_circuits = []
    for qc in circuits:
        noisy_qc = QuantumCircuit()
        for qreg in qc.qregs:
            noisy_qc.add_register(qreg)
        for creg in qc.cregs:
            noisy_qc.add_register(creg)

        for gate in qc:
            g = gate.operation.name
            num_q = gate.operation.num_qubits
            qubits = [q._index for q in gate.qubits]
            pre_error = g != "measure"
            # add gate if it needs to go before the error
            
            if g!='barrier':
                if pre_error:
                    noisy_qc.append(gate)

                # error probability
                if num_q==1:
                    if g == 'reset':
                        error_prob = noise_dict[qubits[0]]['init']
                    elif g == 'measure':
                        error_prob = noise_dict[qubits[0]]['RO']
                    elif g == 'quantum_channel':
                        #quantum channel for idling should be in the circuit even if the probability is set to zero
                        error_prob = noise_dict[qubits[0]]['idle']
                    else:
                        error_prob = noise_dict[qubits[0]]['gate']
                elif num_q==2:
                    error_prob = noise_dict[qubits[0]]['2-gate'][qubits[1]]

                # then adding the error
                if g=='reset' or g=='measure':
                    noisy_qc.append(pauli_error([("I",1-error_prob),("X",error_prob)]), qubits)
                else:
                    noisy_qc.append(depolarizing_error(param=error_prob,num_qubits=num_q), qubits)
                # add gate if it needs to go after the error
                if not pre_error:
                    if not g.startswith("idle_"):
                        noisy_qc.append(gate)

            noisy_circuits.append(noisy_qc)

    if single_circuit:
        noisy_circuits = noisy_circuits[0]

    return noisy_circuits

def transDD(circ, backend, echo="X", echo_num=2, qubit_list=[]):
    """
    Args:
        circ: QuantumCircuit object
        backend (qiskit.providers.ibmq.IBMQBackend): Backend to transpile and schedule the
        circuits for. The numbering of the qubits in this backend should correspond to
        the numbering used in `self.links`.
        echo: gate sequence (expressed as a string) to be used on the qubits. Valid strings 
        are `'X'` and `'XZX'`.
        echo_num: Number of times to repeat the sequence for qubits.
    Returns:
        transpiled_circuit: As `self.circuit`, but with the circuits scheduled, transpiled and
        with dynamical decoupling added.
    """

    initial_layout = []
    initial_layout += [
        circ.qubits[q] for q in range(circ.num_qubits)
    ]

    # transpile to backend and schedule
    circuit = transpile(
        circ, backend, initial_layout=initial_layout,scheduling_method="alap"
    )

    #then dynamical decoupling if needed
    if echo_num:

        # set up the dd sequences
        spacings = []
        if echo == "X":
            dd_sequences = [XGate()] * echo_num
            spacings.append(None)
        elif echo == "XZX":
            dd_sequences = [XGate(), RZGate(np.pi), XGate()] * echo_num
            d = 1.0 / (2 * echo_num - 1 + 1)
            spacing = [d / 2] + ([0, d, d] * echo_num)[:-1] + [d / 2]
            for _ in range(2):
                spacing[0] += 1 - sum(spacing)
            spacings.append(spacing)
        else:
            dd_sequences.append(None)
            spacings.append(None)

        # add in the dd sequences
        durations = InstructionDurations().from_backend(backend)
        if dd_sequences[0]:
            if echo_num:
                if qubit_list == []:
                    qubits = circ.qubits
                else:
                    qubits = qubit_list
            else:
                qubits = None
            pm = PassManager([ALAPScheduleAnalysis(durations),
                  PadDynamicalDecoupling(durations, dd_sequences,qubits=qubits)])
            circuit = pm.run(circuit)

        # make sure delays are a multiple of 16 samples, while keeping the barriers
        # as aligned as possible
        total_delay = [{q: 0 for q in circuit.qubits} for _ in range(2)]
        for gate in circuit.data:
            if gate[0].name == "delay":
                q = gate[1][0]
                t = gate[0].params[0]
                total_delay[0][q] += t
                new_t = 16 * np.ceil((total_delay[0][q] - total_delay[1][q]) / 16)
                total_delay[1][q] += new_t
                gate[0].params[0] = new_t

        # transpile to backend and schedule again
        circuit = transpile(circuit, backend, scheduling_method="alap")

    return circuit

def string2detections(job_result, det_measinds, log_measinds):
    meas_samples = []
    freqs = []
    for string,freq in job_result.items():
        meas_sample = np.array([int(char) for char in string[::-1]])
        meas_samples.append(meas_sample)
        freqs.append(freq)
    
    detector_samples = []
    for sample in meas_samples:
        new_det_sample = []
        for detector in det_measinds:
            new_det_sample.append(sample[detector].sum()%2)
        for log_measind in log_measinds:
            new_det_sample.append(sample[log_measind].sum()%2)
        detector_samples.append(new_det_sample)
    detector_samples = np.array(detector_samples)
    
    return freqs,detector_samples

def get_stim_circuits(
    circuit,
    detectors = None,
    logicals = None,
):
    """Converts compatible qiskit circuits to stim circuits.
       Dictionaries are not complete. For the stim definitions see:
       https://github.com/quantumlib/Stim/blob/main/doc/gates.md
    Args:
        circuit: Compatible gates are Paulis, controlled Paulis, h, s,
        and sdg, swap, reset, measure and barrier. Compatible noise operators
        correspond to a single or two qubit pauli channel.
        detectors: A list of measurement comparisons. A measurement comparison
        (detector) is either a list of measurements given by a the name and index
        of the classical bit or a list of dictionaries, with a mandatory clbits
        key containing the classical bits. A dictionary can contain keys like
        'qubits', 'time', 'basis' etc.
        logicals: A list of logical measurements. A logical measurement is a
        list of classical bits whose total parity is the logical eigenvalue.
        Again it can be a list of dictionaries.

    Returns:
        stim_circuits, stim_measurement_data
    """

    if detectors is None:
        detectors = [{}]
    if logicals is None:
        logicals = [{}]

    if len(detectors) > 0 and isinstance(detectors[0], List):
        detectors = [{"clbits": det, "qubits": [], "time": 0} for det in detectors]

    if len(logicals) > 0 and isinstance(logicals[0], List):
        logicals = [{"clbits": log} for log in logicals]

    stim_circuits = []
    stim_measurement_data = []
    if isinstance(circuit, QuantumCircuit):
        circuit = [circuit]
    for circ in circuit:
        stim_circuit = StimCircuit()

        qiskit_to_stim_dict = {
            "id": "I",
            "x": "X",
            "y": "Y",
            "z": "Z",
            "h": "H",
            "s": "S",
            "sdg": "S_DAG",
            "cx": "CX",
            "cy": "CY",
            "cz": "CZ",
            "swap": "SWAP",
            "reset": "R",
            "measure": "M",
            "barrier": "TICK",
        }
        pauli_error_1_stim_order = {
            "id": 0,
            "I": 0,
            "X": 1,
            "x": 1,
            "Y": 2,
            "y": 2,
            "Z": 3,
            "z": 3,
        }
        pauli_error_2_stim_order = {
            "II": 0,
            "IX": 1,
            "IY": 2,
            "IZ": 3,
            "XI": 4,
            "XX": 5,
            "XY": 6,
            "XZ": 7,
            "YI": 8,
            "YX": 9,
            "YY": 10,
            "YZ": 11,
            "ZI": 12,
            "ZX": 13,
            "ZY": 14,
            "ZZ": 15,
        }

        measurement_data = []
        qreg_offset = {}
        creg_offset = {}
        prevq_offset = 0
        prevc_offset = 0
        for inst, qargs, cargs in circ.data:
            for qubit in qargs:
                if qubit._register.name not in qreg_offset:
                    qreg_offset[qubit._register.name] = prevq_offset
                    prevq_offset += qubit._register.size
            for bit in cargs:
                if bit._register.name not in creg_offset:
                    creg_offset[bit._register.name] = prevc_offset
                    prevc_offset += bit._register.size

            qubit_indices = [
                qargs[i]._index + qreg_offset[qargs[i]._register.name] for i in range(len(qargs))
            ]

            if isinstance(inst, QuantumChannelInstruction):
                qerror = inst._quantum_error
                pauli_errors_types = qerror.to_dict()["instructions"]
                pauli_probs = qerror.to_dict()["probabilities"]
                if pauli_errors_types[0][0]["name"] in pauli_error_1_stim_order:
                    probs = 4 * [0.0]
                    for pind, ptype in enumerate(pauli_errors_types):
                        probs[pauli_error_1_stim_order[ptype[0]["name"]]] = pauli_probs[pind]
                    stim_circuit.append("PAULI_CHANNEL_1", qubit_indices, probs[1:])
                elif pauli_errors_types[0][0]["params"][0] in pauli_error_2_stim_order:
                    # here the name is always 'pauli' and the params gives the Pauli type
                    probs = 16 * [0.0]
                    for pind, ptype in enumerate(pauli_errors_types):
                        probs[pauli_error_2_stim_order[ptype[0]["params"][0]]] = pauli_probs[pind]
                    stim_circuit.append("PAULI_CHANNEL_2", qubit_indices, probs[1:])
                else:
                    raise Exception("Unexpected operations: " + str([inst, qargs, cargs]))
            else:
                # Gates and measurements
                if inst.name in qiskit_to_stim_dict:
                    if len(cargs) > 0:  # keeping track of measurement indices in stim
                        measurement_data.append([cargs[0]._register.name, cargs[0]._index])

                    if qiskit_to_stim_dict[inst.name] == "TICK":  # barrier
                        stim_circuit.append("TICK")
                    elif inst.condition is not None:  # handle c_ifs
                        if inst.name in "xyz":
                            if inst.condition[1] == 1:
                                clbit = inst.condition[0]
                                stim_circuit.append(
                                    qiskit_to_stim_dict["c" + inst.name],
                                    [
                                        StimTarget_rec(
                                            measurement_data.index(
                                                [clbit._register.name, clbit._index]
                                            )
                                            - len(measurement_data)
                                        ),
                                        qubit_indices[0],
                                    ],
                                )
                            else:
                                raise Exception(
                                    "Classically controlled gate must be conditioned on bit value 1"
                                )
                        else:
                            raise Exception(
                                "Classically controlled " + inst.name + " gate is not supported"
                            )
                    else:  # gates/measurements acting on qubits
                        stim_circuit.append(qiskit_to_stim_dict[inst.name], qubit_indices)
                else:
                    raise Exception("Unexpected operations: " + str([inst, qargs, cargs]))

        if detectors != [{}]:
            for det in detectors:
                stim_record_targets = []
                for reg, ind in det["clbits"]:
                    stim_record_targets.append(
                        StimTarget_rec(measurement_data.index([reg, ind]) - len(measurement_data))
                    )
                if det["time"] != []:
                    stim_circuit.append(
                        "DETECTOR", stim_record_targets, det["qubits"] + [det["time"]]
                    )
                else:
                    stim_circuit.append("DETECTOR", stim_record_targets, [])
        if logicals != [{}]:
            for log_ind, log in enumerate(logicals):
                stim_record_targets = []
                for reg, ind in log["clbits"]:
                    stim_record_targets.append(
                        StimTarget_rec(measurement_data.index([reg, ind]) - len(measurement_data))
                    )
                stim_circuit.append("OBSERVABLE_INCLUDE", stim_record_targets, log_ind)

        stim_circuits.append(stim_circuit)
        stim_measurement_data.append(measurement_data)

    return stim_circuits, stim_measurement_data
    
def get_noise_dict_from_backend(backend, date=None):
    '''
    Takes a given backend and the date and returns a noise dictionary based on calibration data.
    '''
    # Pre-reqs
    qubits = range(backend.num_qubits)
    noise_dict = {}
    round_time = 2*np.mean([backend.properties().readout_length(q) for q in range(backend.num_qubits)]) #readout+reset
    properties = backend.properties(datetime=date)

    # Initializing each qubit with single qubit noise
    for qubit in qubits:
        # Defining ROI error
        ROI_error = properties.readout_error(qubit)

        # Defining idle error
        t1 = properties.t1(qubit)
        t2 = properties.t2(qubit)
        t_time = min(t1, t2)
        idle_error = 1 - np.exp(-round_time / t_time)

        # Updating dictionary
        noise_dict[qubit] = {}
        noise_dict[qubit]['init'] = ROI_error
        noise_dict[qubit]['idle'] = idle_error
        noise_dict[qubit]['RO'] = ROI_error
        noise_dict[qubit]['gate'] = 1e-4
        noise_dict[qubit]['2-gate'] = {}#{'default': 1e-2}
        
    # Two qubit gate noise
    all_qiskit_indexes = qubits
    for pair in backend.coupling_map:
        # Making sure connection is in circuit
        if pair[0] in all_qiskit_indexes and pair[1] in all_qiskit_indexes:
            # Getting two qubit gate error
            if backend.name == 'ibm_torino':
                two_gate_error = properties.gate_error('cz', pair)
            else:
                # properties.gate_error('ecr', pair)
                try:
                    two_gate_error = properties.gate_property(gate='ecr')[pair]['gate_error'][0]
                except KeyError:
                    two_gate_error = properties.gate_property(gate='ecr')[pair[::-1]]['gate_error'][0]

            if two_gate_error > .5:
                two_gate_error = .5
            # Updating dictionary 
            noise_dict[pair[0]]['2-gate'][pair[1]] = two_gate_error
            noise_dict[pair[1]]['2-gate'][pair[0]] = two_gate_error

    return noise_dict

### plotting and fitting

def plot_3d_threshold(LogX_fail_3d_d_p,LogZ_fail_3d_d_p,ax,
                      truncate=True,alpha=1,cmap='copper',interpolate=True, colbar = None):
    pG_list = []
    pT_list = []
    pR_list = []
    pth_list = []
    pG_listB = []
    pT_listB = []
    pR_listB = []
    pth_listB = []
    DL_list = []
    DL_listB = []
    for rowind in range(len(LogX_fail_3d_d_p)):
        for columnind in range(len(LogX_fail_3d_d_p[rowind])):
            theta, phi, p_threshold1, _, DL1 = LogX_fail_3d_d_p[rowind][columnind]
            theta, phi, p_threshold2, _, DL2 = LogZ_fail_3d_d_p[rowind][columnind]
            if min(p_threshold1,p_threshold2)==0:
                p_threshold1 = max(p_threshold1,p_threshold2)
                p_threshold2 = max(p_threshold1,p_threshold2)
            p_threshold = min(p_threshold1,p_threshold2)
            pR = p_threshold*np.sin(theta)
            pm = pR
            if p_threshold>0 and 0<theta<0.499*np.pi and 0<phi<0.499*np.pi and (not truncate or p_threshold*np.cos(theta)*np.sin(phi) < .08):
                pG_list.append(p_threshold*np.cos(theta)*np.cos(phi))
                pT_list.append(p_threshold*np.cos(theta)*np.sin(phi))
                pR_list.append(pm)
                pth_list.append(np.sqrt((p_threshold*np.cos(theta)*np.cos(phi))**2+(p_threshold*np.cos(theta)*np.sin(phi))**2+pm**2))
                DL_list.append((DL1+DL2)/2)
            elif p_threshold>0 and (not truncate or p_threshold*np.cos(theta)*np.sin(phi) < 8):
                pG_listB.append(p_threshold*np.cos(theta)*np.cos(phi))
                pT_listB.append(p_threshold*np.cos(theta)*np.sin(phi))
                pR_listB.append(pm)
                pth_listB.append(np.sqrt((p_threshold*np.cos(theta)*np.cos(phi))**2+(p_threshold*np.cos(theta)*np.sin(phi))**2+pm**2))
                DL_listB.append((DL1+DL2)/2)

    pG_list.extend(pG_listB)
    pT_list.extend(pT_listB)
    pR_list.extend(pR_listB)
    pth_list.extend(pth_listB)
    DL_list.extend(DL_listB)


    xi = np.linspace(0, max(pG_list), 200)
    yi = np.linspace(0, max(pT_list), 200)

    minn, maxx = min(DL_list), max(DL_list)
    norm = colors.Normalize(minn, maxx,clip=False)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    if interpolate:
        Zi = griddata((pG_list, pT_list), pR_list, (xi[None, :], yi[:, None]), method='linear')
        Zi[np.isnan(Zi)]=0
        DLi = griddata((pG_list, pT_list), DL_list, (xi[None, :], yi[:, None]), method='linear')
        Xi,Yi = np.meshgrid(xi,yi)
        color_dimension = DLi
        m.set_array([])
        fcolors = m.to_rgba(color_dimension)
        for i in range(len(fcolors)):
            for j in range(len(fcolors)):
                if Zi[i,j]==0:
                    fcolors[i,j] = (1,1,1,1)
                    Zi[i,j] = np.nan
        ax.plot_surface(Xi,Yi,Zi,rstride=1,cstride=1, facecolors=fcolors, vmin=minn, vmax=maxx, alpha = alpha, antialiased = False, linewidth=0, shade=False)
        if colbar:
            print(minn,maxx)
            cbar = plt.colorbar(m, ax=ax, shrink = 0.6)
            cbar.set_label(r'detector likelihood, $\langle D \rangle$', rotation=270,labelpad=13.0)#, fontfamily = 'times')
            ax.scatter(pT_list,pG_list,pR_list, c=DL_list, cmap=cmap,marker='o',alpha=1,s=20)
        ax.scatter(pT_list,pG_list,pR_list, c=DL_list, cmap=cmap,marker='o',alpha=1,s=20)
    else:
        # ax.scatter(pG_list,pT_list,pR_list, c=pth_list, cmap=cmap,marker='o', alpha= alpha)
        ax.scatter(pT_list,pG_list,pR_list, c=DL_list, cmap=cmap,marker='o', alpha= 1)
        minn, maxx = min(DL_list), max(DL_list)
        norm=colors.Normalize(minn, maxx,clip=True)
        if colbar:
            cbar = plt.colorbar(m, ax=ax, shrink = 0.6)
            cbar.set_label(r'detector likelihood, $\langle D \rangle$', rotation=270,labelpad=13.0)#, fontfamily = 'times')

    
    # ax.set_xlabel('$p_G\ [\%]$')
    # ax.set_ylabel('$p_T\ [\%]$')
    ax.set_ylabel(r'$p_G\, [\%]$')
    ax.set_xlabel(r'$p_T\, [\%]$')
    ax.set_zlabel(r'$p_{RR}\, [\%]$')

    # ax.set_xticks(np.arange(0,1.8,0.3))
    # ax.set_xlim(0,)
    # ax.set_ylim(0,)
    # ax.set_zlim(1e-5,)
    ax.view_init(elev=20., azim=20)

def fit_plane(LogX_fail_3d_d_p,LogZ_fail_3d_d_p,rescalepR = True):
    theta_phi_pth_list = []
    pG_list = []
    pT_list = []
    pR_list = []
    for rowind in range(len(LogX_fail_3d_d_p)):
        for columnind in range(len(LogX_fail_3d_d_p[rowind])):
            theta, phi, p_threshold1, _ = LogX_fail_3d_d_p[rowind][columnind]
            theta, phi, p_threshold2, _ = LogZ_fail_3d_d_p[rowind][columnind]
            if min(p_threshold1,p_threshold2)==0:
                p_threshold1 = max(p_threshold1,p_threshold2)
                p_threshold2 = max(p_threshold1,p_threshold2)
            p_threshold = min(p_threshold1,p_threshold2)
            pR = p_threshold*np.sin(theta)
            pm = pR
            if rescalepR:    
                pm = sum([comb(2,i)*((8*pR/15)**i)*((1-8*pR/15)**(2-i))*(pR**j)*((1-pR)**(1-j)) for i in range(3) for j in range(2) if i+j%2])
            if p_threshold>0: #and 0<theta<0.499*np.pi and 0<phi<0.499*np.pi:
                pG,pT = [p_threshold*np.cos(theta)*np.cos(phi),p_threshold*np.cos(theta)*np.sin(phi)]
                pG_list.append(pG*100)
                pT_list.append(pT*100)
                pR_list.append(pm*100)
                new_theta = np.arctan2(pm,np.sqrt(pG**2+pT**2))
                new_phi = np.arctan2(pT,pG)
                theta_phi_pth_list.append((new_theta,new_phi,np.sqrt(pG**2+pT**2+pm**2)*100))

    A = np.array(theta_phi_pth_list)

    def func(theta_phi, pGth, pTth, pRth):
        theta,phi = theta_phi.transpose()
        return 1./(np.cos(theta)*np.cos(phi)/pGth + np.cos(theta)*np.sin(phi)/pTth+ np.sin(theta)/pRth)

    guess = (max(pG_list),max(pT_list),max(pR_list))
    params, pcov = optimize.curve_fit(func, A[:,:2], A[:,2], guess)
    fit_error_list = []
    for theta,phi,p_threshold in theta_phi_pth_list:
        if p_threshold>0: #and 0<theta<0.499*np.pi and 0<phi<0.499*np.pi:
            fit_error_list.append(1-func(np.array([theta,phi]),params[0],params[1],params[2])/p_threshold)
    fit_error_list = np.array(fit_error_list)
    return [params,np.sqrt(pcov.diagonal()),np.sqrt(np.mean(fit_error_list**2)),max(fit_error_list)]