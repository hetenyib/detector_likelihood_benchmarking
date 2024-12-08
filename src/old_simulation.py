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
import matplotlib.pyplot as plt
import beliefmatching as bm

from hex_code import Hex_Code
import planar_codes.layout_templates as templates
from planar_codes.CSS import CSS as planar_css
from planar_codes.CSS_direct import CSS_direct



from qiskit_code.result_processing import get_string_heatmap
from qiskit_code.qiskit_glue import Cross_Platform_Code

from time import perf_counter
from scipy.optimize import curve_fit
from qiskit import transpile
from qiskit_aer import AerSimulator


def run_aer_sim(qiskit_circuit, shots=1024):
    backend = AerSimulator()
    qc_compiled = transpile(qiskit_circuit, 
                            backend)

    # Run qiskit sim
    job_sim = backend.run(qc_compiled, shots=shots)
    result_sim = job_sim.result()
    counts = result_sim.get_counts(qc_compiled)

    return counts
# End run_aer_sim


def log_er_per_round(sub_rounds, E_logical_round):
    return (1 - (1 - 2 * E_logical_round) ** sub_rounds) / 2
# End log_er_per_round

def log_er_distance(distance, C, lamb):
    return C / (lamb ** ((distance + 1) / 2))
# End log_er_distance



def lambda_factor(noise=1e-3*np.ones(5), 
                  noise_dict=None,
                 shots=int(1e4), 
                 distances=[12,8,4],
                 rounds=np.arange(12, 120, 12),
                 style='css',
                 geometry='planar',
                 plot=True,
                 title='Logical error rate per round',
                 detector_type='original',
                 border='direct',
                 pauli_cycle=0,
                 basis='primary'):
    '''
    Simulates varying distances for a given physical error rate.
    Calculates lambda factor from results. 
    '''
    # Simulation
    error_rates = []
    for d in distances:
        distance_rates = []
        for num_rounds in rounds:
            if geometry == 'toric':
                width = distance // 4
                length = distance // 2
                layout = Toric_Layout(width, length, noise=noise, layout_type=1, pauli_cycle=pauli_cycle)
            else:
                size = 3 * (d // 4) - 1
                layout = templates.parallelogram(size, size, noise=noise, noise_dict=noise_dict, basis=basis, border=border)
            code = Hex_Code(layout, sub_rounds=int(num_rounds), style=style)
            distance_rates.append(error_rate_sim(code, shots))
            print('.', end='')
        error_rates.append(distance_rates)
        print(',',end='')
    print('!')

    # Process Data, first the error rate as a function of sub_rounds
    if plot:
        plt.figure(figsize=(12,6))
        plt.subplot(121)
    colors=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    distance_rates = []
    for d in range(len(distances)):
        rates, distance = error_rates[d], distances[d]
        round_error_rate = 1 - curve_fit(log_er_per_round, rounds, rates)[0][0]
        if plot:
            plt.plot(rounds, rates, 'o', label=f'd={distance}', color=colors[d])
            plt.plot(rounds, log_er_per_round(rounds, round_error_rate), '--')
        distance_rates.append(round_error_rate)

    if plot:
        plt.legend()
        plt.title('')
        plt.ylabel('Logical error')
        plt.xlabel('Sub-rounds')
        plt.title(title)

    # Now to get lambda factor from the distance error rates per round
    C, lamb = curve_fit(log_er_distance, distances, distance_rates)[0]
    if plot:
        plt.subplot(122)
        est = [log_er_distance(d, C, lamb) for d in distances]
        plt.plot(distances, distance_rates, 'o')
        plt.plot(distances, est, '--')
        plt.yscale('log')
        plt.title(f'Lambda: {lamb}')
        plt.grid(which='both')
        plt.ylabel('Logical error rate per sub_round')
        plt.xlabel('Distance')
        plt.show()

    return error_rates, distance_rates, lamb
# End lambda_factor


def error_rate_sim(code:Hex_Code, shots):
    '''
    Runs an error rate simulation for a given Hex_Code object.
    '''
    #time_0 = perf_counter()
    syns, obs = code.sample(shots)
    #time_1 = perf_counter()
    errors = 0
    for s in range(shots):
        errors += code.match.decode(syns[s]) != obs[s]
    #time_2 = perf_counter()
    #print('Sampling time:', time_1 - time_0, '; Decoding time:', time_2 - time_1, 'fraction d/s-', (time_2 - time_1)/(time_1 - time_0))
    return errors[0] / shots
# End error_rate_sim


def bm_error_rate_sim(code:Hex_Code, shots):
    '''
    Runs an error rate simulation for a given Hex_Code object.
    '''
    #time_0 = perf_counter()
    syns, obs = code.sample(shots)
    
    belief = bm.BeliefMatching(code.circuit, max_bp_iters=20)

    predicted_observables = belief.decode_batch(syns)
    num_mistakes = np.sum(np.any(predicted_observables != obs, axis=1))

    #print('Sampling time:', time_1 - time_0, '; Decoding time:', time_2 - time_1, 'fraction d/s-', (time_2 - time_1)/(time_1 - time_0))
    return num_mistakes / shots
# End error_rate_sim


def quick_toric_threshold(distances=[12, 8, 4],
                    noises='light',
                    shots=1000,
                    sub_rounds=30,
                    noise_breakdown=np.ones(5),
                    code_type=0, 
                    init_noise=False, 
                    ro_noise=False,
                    title='Threshold Graph',
                    detector_type='original',
                    pauli_cycle=0):
    
    if noises == 'light':
        noises = [1e-2, 5.5e-3, 2e-3, 1e-3, 3.3e-4, 1e-4]
    elif noises == 'detailed':
        noises = [1e-2, 5.5e-3, 3.3e-3, 2e-3, 1e-3, 5.5e-4, 3.3e-4, 2e-4, 1e-4]

    data = []
    for distance in distances:
        dats = []
        for n in noises:
            noise = n * np.array(noise_breakdown)
            toric_width = distance // 4
            toric_length = distance // 2
            layout_type = 0 if code_type == 0 else 1
            
            layout = Toric_Layout(toric_width, 
                                  toric_length, 
                                  noise=noise, 
                                  layout_type=layout_type,
                                  pauli_cycle=pauli_cycle)
            if code_type <= 1:
                code = H3(layout, int(sub_rounds), 
                                init_noise=init_noise, 
                                ro_noise=ro_noise)
            elif code_type == 2: 
                code = Double_Rung(layout, int(sub_rounds), 
                                   init_noise=init_noise, 
                                   ro_noise=ro_noise,
                                   detector_type=detector_type)
                
            elif code_type== 3:
                code = CSS(layout, int(sub_rounds), 
                                   init_noise=init_noise, 
                                   ro_noise=ro_noise)
            
            error_rate = error_rate_sim(code, shots)
            dats.append(error_rate)
            print('.', end='')
        data.append(dats)
        print('!', end='')
    print('#')

    plt.clf()
    fig, ax = plt.subplots(1, 1)
    for d in range(len(data)):
        plt.plot(noises, data[d], 'o--', label=f'd={distances[d]}')
    plt.plot(noises, noises, '--', color='lightgray')
    ax.set_ylim(max(min(np.array(data).flatten()), 1/shots), .7)
    ax.set_xlim(min(noises), max(noises))
    ax.loglog()
    ax.set_title(title)
    ax.set_xlabel("Phyical Error Rate")
    ax.set_ylabel("Logical Error Rate per Shot")
    ax.grid(which='major', color='lightgray')
    ax.grid(which='minor', color='lightgray')
    ax.legend()
    fig.set_dpi(120) 
    plt.show()
    return data
# End quick_threshold


def quick_planar_threshold(distances=[16, 12, 8],
                    noises='light',
                    shots=1000,
                    sub_rounds=30,
                    noise_breakdown=np.ones(5),
                    init_noise=False, 
                    ro_noise=False,
                    title='Threshold Graph',
                    border='direct',
                    basis='primary',
                    vline=None,
                    layout_style='diamond'):
    assert not (layout_style == 'diamond' and border == 'dangle'), 'No dangle borders or a diamond layout.'
    if noises == 'light':
        noises = [1e-2, 5.5e-3, 2e-3, 1e-3, 3.3e-4, 1e-4]
    elif noises == 'detailed':
        noises = [1e-2, 5.5e-3, 3.3e-3, 2e-3, 1e-3, 5.5e-4, 3.3e-4, 2e-4, 1e-4]

    data = []
    for distance in distances:
        dats = []
        for n in noises:
            noise = n * np.array(noise_breakdown)
            size = 3 * (distance // 4) - 1
            
            if layout_style == 'parallelogram':
                layout = templates.parallelogram(size, 
                                                size, 
                                                noise=noise, 
                                                border=border,
                                                basis=basis)
            elif layout_style == 'diamond':
                layout = templates.diamond(size, 
                                            noise=noise, 
                                            basis=basis)
            if border == 'dangle':
                code = planar_css(layout, 
                            int(sub_rounds), 
                            init_noise=init_noise, 
                            ro_noise=ro_noise)
            elif border == 'direct':
                code = CSS_direct(layout, 
                            int(sub_rounds), 
                            init_noise=init_noise, 
                            ro_noise=ro_noise)
            else:
                raise NameError('Use "dangle" or "direct" for border type.')

            error_rate = error_rate_sim(code, shots)
            dats.append(error_rate)
            print('.', end='')
        data.append(dats)
        print('!', end='')
    print('#')

    plt.clf()
    fig, ax = plt.subplots(1, 1)
    for d in range(len(data)):
        plt.plot(noises, data[d], 'o--', label=f'd={distances[d]}')
    plt.plot(noises, noises, '--', color='lightgray')
    ax.set_ylim(max(min(np.array(data).flatten()), 1/shots), .7)
    ax.set_xlim(min(noises), max(noises))
    ax.loglog()
    ax.set_title(title)
    ax.set_xlabel("Phyical Error Rate")
    ax.set_ylabel("Logical Error Rate per Shot")
    ax.grid(which='both', color='lightgray')
    ax.legend()
    if vline is not None:
        ax.vlines(vline, 
                  max(min(np.array(data).flatten()), 1/shots), .7, 
                  linestyles='dashed', color='lightgray')
    fig.set_dpi(120) 
    plt.show()
    return data
# End quick_threshold


def simulate_string_heatmap(simulating_hex:(Hex_Code or Cross_Platform_Code), 
                            decoding_hex:(Hex_Code or Cross_Platform_Code),
                            shots:int,
                            printing=False):
    '''
    Gets simulation data and feeds it to get_string_heatmap
    to create a heatmap
    '''
    detectors, log_ops = simulating_hex.sample(shots)

    error_shots = []
    for s in range(shots):
        correction = decoding_hex.match.decode(detectors[s])
        if int(correction) != int(log_ops[s]):
            error_shots.append(s)
    if printing:
        print('Error count:', len(error_shots))
        print('Error rate:', len(error_shots) / shots)

    heatmap = get_string_heatmap(decoding_hex, 
                                 detectors, 
                                 error_shots,
                                 printing=printing)
    
    return heatmap
# End simulate_string_heatmap
