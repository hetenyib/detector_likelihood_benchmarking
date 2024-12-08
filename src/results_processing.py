# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import json
import numpy as np
import os
from scipy.optimize import curve_fit
import warnings

from src.qiskit_glue import Cross_Platform_Code, add_calibration_noise
import src.layout_templates as templates
from src.CSS_direct import Hex_Code
from src.layout_planar_main import index_from_coord
from src.effective_p import *
from src.simulation import error_rate_sim


##########################################
#   Reading Files and getting Results    #
##########################################


def get_existing_runs():
    runs = []
    for file_name in os.listdir('runs/'):
        if file_name != 'old' and file_name != '.DS_Store':
            # Getting data
            entries = file_name.split('_')
            data = {entry.split('-')[0]:entry.split('-')[1] for entry in entries}

            # Chop off .txt from end
            data['id'] = data['id'][:-4]
            runs.append(data)
    return runs
# End get_existing_runs


def list_runs(date=None,
              device=None,
              layout=None,
              size=None,
              location=None,
              subrounds=None,
              basis=None,
              border=None,
              shots=None,
              DD=None,
              id=None,):
    runs = get_existing_runs()

    runs_subset = [run for run in runs 
                if (
                    (date is None or run['date'] in date)
                and (device is None or run['device'] in device)
                and (layout is None or run['layout'] in layout)
                and (size is None or run['size'] in size)
                and (location is None or run['location'] in location)
                and (subrounds is None or run['subrounds'] in subrounds)
                and (basis is None or run['basis'] in basis)
                and (border is None or run['border'] in border)
                and (shots is None or run['shots'] in shots)
                and (DD is None or run['DD'] in DD)
                and (id is None or run['id'] in id)
                )]
    
    return runs_subset
# End list_runs


def get_cross_code(entry:dict, noise_model='median'):
    '''
    Takes a single dictionary produced by list_runs or get_existing_runs
    and returns the associated Cross_Platform_Code object.
    '''
    # Defining noise # TODO: DOUBLE CHECK IDLE NOISE, LOOKS ORDER OF MAG TOO HIGH!
    if noise_model == 'median':
        if entry["device"] == 'sherbrooke':
            noises = [1e-2, 2e-3, 1e-2, 1e-3, 7e-3]
        elif entry['device'] == 'nazca':
            noises = [2.5e-2, 3.85e-3, 2.5e-2, 1e-4, 1e-2]
        elif entry['device'] == 'cusco':
            noises = [3.15e-2, 5e-3, 3.14e-2, 1e-4, 1.45e-2] 
        elif entry['device'] == 'brisbane':
            noises = [1.35e-2, 3.3e-3, 1.35e-2, 1e-4, 7.5e-3]
        elif entry['device'] == 'osaka':
            noises = [2e-2, 1e-2, 2e-1, 1e-4, 6.67e-3]
        elif entry['device'] == 'kyoto':
            noises = [1.3e-2, 1.6e-2, 1.3e-2, 1e-4, 7.78e-2]
        elif entry['device'] == 'torino':
            noises = [1.89e-2, 3e-3, 1.89e-2, 1e-4, 3.26e-3]
        else:
            raise KeyError(f'{entry["device"]} does not have a noise breakdown associated.')
        
    elif noise_model == 'flat':
        noises = 1e-3 * np.ones(5)

    else:
        raise ValueError('Only "flat" or "median" are accepted noise models. To add calibration data use "flat" and method add_calibration_data() from qiskit_glue.')
    

    # Defining layout and hex code
    layout = templates.diamond(int(entry['size']), 
                                noise=noises, 
                                basis=entry['basis'])
    hex_code = Hex_Code(layout, int(entry['subrounds']))

    
    # Defining cross_code
    if (entry['device'] == 'sherbrooke'
     or entry['device'] == 'nazca'
     or entry['device'] == 'cusco'
     or entry['device'] == 'brisbane'
     or entry['device'] == 'osaka'
     or entry['device'] == 'kyoto'):
        device_size = 127
    elif entry['device'] == 'torino':
        device_size = 133
    elif entry['device'] == 'auckland':
        device_size = 27
    elif entry['device'] == 'seattle':
        device_size = 433
    else:
        raise KeyError(f'{entry["device"]} is not a listed device in get_cross_code()')
    
    cross_code = Cross_Platform_Code(hex_code, 
                                     device_size, 
                                     int(entry['location']),
                                     generic_reset=True)

    return cross_code
# End get_cross_code


def get_run_file_name(entry:dict) -> str:

    file_name = f'runs/'
    file_name += f'date-{entry["date"]}'
    file_name += f'_device-{entry["device"]}'
    file_name += f'_layout-{entry["layout"]}'
    file_name += f'_size-{entry["size"]}'
    file_name += f'_location-{entry["location"]}'
    file_name += f'_subrounds-{entry["subrounds"]}'
    file_name += f'_basis-{entry["basis"]}'
    file_name += f'_border-{entry["border"]}'
    file_name += f'_shots-{entry["shots"]}'
    file_name += f'_DD-{entry["DD"]}'
    file_name += f'_id-{entry["id"]}'
    file_name += '.txt'

    return file_name
# End get_run_file_name



def results_from_file(file_name, code:Cross_Platform_Code):
    '''
    Reads file with counts information in it and returns 
    arrays with detector data and logical operator readout.
    Returns detectors followed by logical observables.
    '''
    # Getting info
    counts = json.load(open(file_name))

    detectors, logical_ops = [], []
    for count in counts:
        dets = code.dtrack.get_detector_values(count)
        log_ops = code.dtrack.get_log_op(count)
        # Appent a case for each shot of same result
        for _ in range(counts[count]):
            detectors.append(dets)
            logical_ops.append(log_ops)
    return detectors, logical_ops
# End results_from_file


def results_from_counts(counts, code:Cross_Platform_Code):
    '''
    Reads file with counts information in it and returns 
    arrays with detector data and logical operator readout.
    Returns detectors followed by logical observables.
    '''
    detectors, logical_ops = [], []
    for count in counts:
        dets = code.dtrack.get_detector_values(count)
        log_ops = code.dtrack.get_log_op(count)
        # Appent a case for each shot of same result
        for _ in range(counts[count]):
            detectors.append(dets)
            logical_ops.append(log_ops)
    return detectors, logical_ops
# End results_from_file


def get_error_rate_from_counts(counts, code:Cross_Platform_Code):
    '''
    Given a filename, which contains the results of a run of 
    the specifid code, returns an error rate using he result_from_file method.
    '''
    detectors, logical_ops = results_from_counts(counts, code)
    shots = len(detectors)
    assert len(detectors) == len(logical_ops), 'detectors and logical ops contain different # of shots.'

    errors = 0
    for s in range(shots):
        correction = code.match.decode(detectors[s])
        if correction != logical_ops[s]:
            errors += 1

    return errors / shots
# End get_error_rate


def get_error_rate_from_file(file_name, code:Cross_Platform_Code):
    '''
    Given a filename, which contains the results of a run of 
    the specifid code, returns an error rate using he result_from_file method.
    '''
    detectors, logical_ops = results_from_file(file_name, code)
    shots = len(detectors)
    assert len(detectors) == len(logical_ops), 'detectors and logical ops contain different # of shots.'

    errors = 0
    for s in range(shots):
        correction = code.match.decode(detectors[s])
        if correction != logical_ops[s]:
            errors += 1

    return errors / shots
# End get_


def error_rate_from_results(code:(Cross_Platform_Code or Hex_Code), detectors, logical_ops):
    '''
    Takes detectors, logical ops, and a code (for access to a matching graph)
    and returns an error rate.
    '''
    shots = len(detectors)
    assert len(detectors) == len(logical_ops), 'detectors and logical ops contain different # of shots.'

    errors = 0
    for s in range(shots):
        correction = code.match.decode(detectors[s])
        if correction != logical_ops[s]:
            errors += 1

    return errors / shots
# End get_


def gather_run_data(date=None,
              device=None,
              layout=None,
              size=None,
              location=None,
              subrounds=None,
              basis=None,
              border=None,
              shots=None,
              DD=None,
              id=None,
              file_name='Results/first_pass_data.json'):
    '''
    Returns all run entries from given file_name given constraints 
    passed as kwargs. No kwargs will return every run in the file.
    '''
    
    data = json.load(open(file_name))

    runs_to_return = []
    for key in data:
        run = data[key]

        if ((date is None or run['date'] in date)
            and (device is None or run['device'] in device)
            and (layout is None or run['layout'] in layout)
            and (size is None or run['size'] in size)
            and (location is None or run['location'] in location)
            and (subrounds is None or run['subrounds'] in subrounds)
            and (basis is None or run['basis'] in basis)
            and (border is None or run['border'] in border)
            and (shots is None or run['shots'] in shots)
            and (DD is None or run['DD'] in DD)
            and (id is None or run['id'] in id)):
            runs_to_return.append(run)

    return runs_to_return
# End gather_run_data


def process_run(run,
                with_truncs=False):
    '''
    Takes a run dictionary with description of the data, then 
    looks up the file of raw output from that. It created the detectors 
    and logical operators from the raw counts, then finds error rates
    as well as detector likelihood for each anyon type. 
    Returns dictionary
    '''
    # Getting data (Defaulting to median decoding)
    cross_code = get_cross_code(run)
    detectors, logical_ops = results_from_file(get_run_file_name(run), cross_code)

    # Getting Detectors
    e_avg, m_avg = get_em_averages(cross_code, detectors, 
                                   include_spatial_truncs=with_truncs)

    # Error rate
    shots = len(detectors)
    if int(run['shots']) != shots:
        warnings.warn(f'Descrepency in {run} for number of shots.')

    errors = 0
    for s in range(shots):
        correction = cross_code.match.decode(detectors[s])
        if correction != logical_ops[s]:
            errors += 1
    error_rate = errors / shots

    # Structuring data
    data_dict = {'date': run['date'],
                'device': run['device'],
                'size': int(run['size']),
                'location': int(run['location']),
                'subrounds': int(run['subrounds']),
                'basis': run['basis'],
                'DD': run['DD'],
                'shots': shots,
                'error_rate': error_rate,
                'e_likelihood': e_avg,
                'm_likelihood': m_avg}
    
    return data_dict
# End process_run


def get_error_rate_per_round_data(date=None,
                                    device=None,
                                    size=None,
                                    location=None,
                                    basis=None,
                                    DD=None,
                                    file_name = 'all_run_data.json'):
    '''
    Sloppy approach, but gets the job done. Gathers all specified 
    data and groups it by parameters. Then gives a fit to it for a 
    real device error rate per sub-round to compare later simulations to.
    '''
    runs = gather_run_data(date=date, device=device, size=size, 
                           location=location, basis=basis, DD=DD,
                           file_name=file_name
                           )
    
    # Group by non-sub_round information
    all_data = {}
    for run in runs:
        # Specifying date
        if run['date'] not in all_data:
            all_data[run['date']] = {}
        date_data = all_data[run['date']]

        # Specifying device
        if run['device'] not in date_data:
            date_data[run['device']] = {}
        device_data = date_data[run['device']]

        # Specifying size
        if run['size'] not in device_data:
            device_data[run['size']] = {}
        size_data = device_data[run['size']]

        # Specifying location
        if run['location'] not in size_data:
            size_data[run['location']] = {}
        location_data = size_data[run['location']]

        # Specifying basis
        if run['basis'] not in location_data:
            location_data[run['basis']] = {}
        basis_data = location_data[run['basis']]

        # Specifying DD
        if run['DD'] not in basis_data:
            basis_data[run['DD']] = {}
        DD_data = basis_data[run['DD']]

        # Adding sub_round Data
        if len([key for key in DD_data]) == 0:
            basis_data[run['DD']] = {'sub_rounds': [], 
                                     'error_rates': [], 
                                     'e_likelihood':[], 
                                     'm_likelihood':[]}
            DD_data = basis_data[run['DD']]

        DD_data['sub_rounds'].append(run['subrounds'])
        DD_data['error_rates'].append(run['error_rate'])
        DD_data['e_likelihood'].append(run['e_likelihood'])
        DD_data['m_likelihood'].append(run['m_likelihood'])

    # Now to calculate fits
    list_results = []
    for date_key in all_data:
        date_dict = all_data[date_key]
        for device_key in date_dict:
            device_dict = date_dict[device_key]
            for size_key in device_dict:
                size_dict = device_dict[size_key]
                for location_key in size_dict:
                    location_dict = size_dict[location_key]
                    for basis_key in location_dict:
                        basis_dict = location_dict[basis_key]
                        for DD_key in basis_dict:
                            DD_dict = basis_dict[DD_key]
                            # Now to the numbers

                            sub_rounds = DD_dict['sub_rounds']
                            error_rates = DD_dict['error_rates']
                            fit, covar = curve_fit(log_er_per_round, sub_rounds, error_rates)
                            DD_dict['round_error_rate'] = 1 - fit[0]
                            DD_dict['fit_error'] = np.sqrt(covar[0][0])
                            DD_dict['shots'] = run['shots']
                            DD_dict['average_detector_likelihood'] = np.average(
                                [lik for lik in DD_dict['e_likelihood'] if lik is not None]
                                + [lik for lik in DD_dict['m_likelihood'] if lik is not None]
                            )
                            
                            data_pack = {'meta_data': {'date': date_key,
                                                        'device': device_key, 
                                                        'size': size_key, 
                                                        'location': location_key, 
                                                        'basis': basis_key,
                                                        'DD': DD_key},
                                         'data': DD_dict}
                            list_results.append(data_pack)

    return list_results
# End get_error_rate_per_round_data


def get_avgs_from_dict(noise_dict):
    '''
    Takes noise dictionary as an argument then returns length 5
    array defining average noise levels: [init, idle, RO, single-, two-qubit gate]
    '''
    init_avg = np.average([noise_dict[qubit]['init'] for qubit in noise_dict])
    idle_avg = np.average([noise_dict[qubit]['idle'] for qubit in noise_dict])
    RO_avg = np.average([noise_dict[qubit]['RO'] for qubit in noise_dict])
    single_avg = 1e-4#np.average([noise_dict[qubit]['init'] for qubit in noise_dict])
    two_gate_avg = np.average([noise_dict[qubit]['2-gate'][connection]
                        for qubit in noise_dict
                        for connection in noise_dict[qubit]['2-gate']
                        if connection != 'default'])
    
    return [init_avg, idle_avg, RO_avg, single_avg, two_gate_avg]
# End get_avgs_from_dict


def get_simulation_comparison(data, backend, shots=int(1e4), 
                              printing=False, fit_index=None,
                              alt=False, only_p=False, date_specific=True):
    '''
    Given a dictionary outputed by get_error_rate_per_round_data(), 
    returns results for 3 different simulation techniques in a dictionary form:
    Average, Calibration, and Effective-p
    '''
    # Setting Up Simulation Work
    if fit_index is None:
        fit_index = max(data['data']['sub_rounds'])
    #index_max = data['data']['sub_rounds'].index(max_sub_rounds)

    if data['meta_data']['basis'] == 'primary':
        det_type = 'e_likelihood'
    else:
        det_type = 'm_likelihood'

    if type(fit_index) == int:
        if alt:
            eff_p = get_effective_p_alt_no_truncs(data['data'][det_type][data['data']['sub_rounds'].index(fit_index)],
                                    data['meta_data']['size'], fit_index)
        else:
            eff_p = get_effective_p_no_truncs(data['data'][det_type][data['data']['sub_rounds'].index(fit_index)],
                                    data['meta_data']['size'], fit_index)
        # Probably should verify this in a notebook.
        # Also compare to the alt strategy. 
    elif type(fit_index) == list:
        if alt:
            eff_p = np.average([get_effective_p_alt_no_truncs(data['data'][det_type][data['data']['sub_rounds'].index(fit)],
                                    data['meta_data']['size'], fit)
                                    for fit in fit_index])
        else:
            eff_p = np.average([get_effective_p_no_truncs(data['data'][det_type][data['data']['sub_rounds'].index(fit)],
                                    data['meta_data']['size'], fit)
                                    for fit in fit_index])
    avg_ers = {}
    cal_ers = {}
    eff_p_ers = {}

    ### Simulation
    if printing:
        print('Siming', end='')
    eff_p_layout = templates.diamond(data['meta_data']['size'], noise=eff_p*np.ones(5))
    for sub_rounds in data['data']['sub_rounds']:
        # p_eff simulation
        device_size = 133 if data['meta_data']['device'] == 'torino' else 127
        p_eff_hex_code = Hex_Code(eff_p_layout, sub_rounds)
        p_eff_cross_code = Cross_Platform_Code(p_eff_hex_code, device_size, data['meta_data']['location'])
        eff_p_ers[sub_rounds] = error_rate_sim(p_eff_cross_code, shots)

        if not only_p:
            # Calibrated simulation
            date = data['meta_data']['date'] if date_specific else None
            calibrated_cross_code = add_calibration_noise(p_eff_cross_code, data['meta_data']['size'], backend, date=date)
            cal_ers[sub_rounds] = error_rate_sim(calibrated_cross_code, shots)
            
            # Average noise simulation
            avg_noise = get_avgs_from_dict(calibrated_cross_code.layout.noise_dict)
            avg_layout = templates.diamond(data['meta_data']['size'], noise=avg_noise)
            avg_hex_code = Hex_Code(avg_layout, sub_rounds)
            avg_ers[sub_rounds] = error_rate_sim(avg_hex_code, shots)
        if printing:
            print('.',end='')
    if printing:
        print(',')

    ### Fitting 
    labels = ['Average', 'Calibrated', 'Effective-p']
    sim_fits = {l:None for l in labels}
    if not only_p:
        # Average
        fit, covar = curve_fit(log_er_per_round, 
                                [key for key in avg_ers], 
                                [avg_ers[key] for key in avg_ers])
        sim_fits['Average'] = {'fit': 1 - fit[0], 'error': np.sqrt(covar[0][0]), 
                            'data': avg_ers, 'shots': shots}
    
        # Calibrated
        fit, covar = curve_fit(log_er_per_round, 
                                [key for key in cal_ers], 
                                [cal_ers[key] for key in cal_ers], 
                                sigma=1/np.sqrt(shots)*np.ones(len(cal_ers)))
        sim_fits['Calibrated'] = {'fit': 1 - fit[0], 'error': np.sqrt(covar[0][0]), 
                                'data': cal_ers, 'shots': shots}

    # Eff-p
    fit, covar = curve_fit(log_er_per_round, 
                            [key for key in eff_p_ers], 
                            [eff_p_ers[key] for key in eff_p_ers])
    sim_fits['Effective-p'] = {'fit': 1 - fit[0], 'error': np.sqrt(covar[0][0]), 
                            'data': eff_p_ers, 'shots': shots}
    
    sim_fits['eff_p'] = eff_p
    
    return sim_fits
# End get_simulation_comparison




#################
#   Heatmaps    #
#################


def get_em_averages(code:(Hex_Code or Cross_Platform_Code),
                    detectors,
                    include_spatial_truncs=True):
    '''
    Similar to get_em_heatmaps, but does not store location data,
    only overall average.
    '''
    if type(code) == Hex_Code:
        coords = code.circuit.get_detector_coordinates()
    elif type(code) == Cross_Platform_Code:
        coords = code.stim_circuit.get_detector_coordinates()

    shots = len(detectors)
    time_max = max([coords[key][2] for key in coords])
    trunc_layers = (0, 1, 2, 3, time_max - 1, time_max)
    layout = code.layout

    m_detectors = []
    e_detectors = []
    for s in range(shots):
        for d in coords:
            # Only if detector triggered
            if coords[d][2] not in trunc_layers:
                m_type = coords[d][3] == 1 if len(coords[d]) == 4 else (coords[d][0] + coords[d][1]) % 4 // 2
                index = index_from_coord(coords[d][0], coords[d][1], layout.width)
                # Bulk Detectors
                if index in layout.plaq_dict:
                    relevent_list = m_detectors if m_type else e_detectors
                    relevent_list.append(detectors[s][d])
                # Spatially truncated detectors
                elif index in layout.truncated_plaquettes and include_spatial_truncs:
                    relevent_list = m_detectors if m_type else e_detectors
                    relevent_list.append(detectors[s][d])
    
    e_avg = np.average(e_detectors) if len(e_detectors) > 0 else None
    m_avg = np.average(m_detectors) if len(m_detectors) > 0 else None

    return e_avg, m_avg
# End get_em_averages



def get_em_heatmaps(code:(Hex_Code or Cross_Platform_Code),
                    detectors,
                    include_spatial_truncs=True,  
                    include_timelike_truncs=True):
    # Gathering heatmap
    if type(code) == Hex_Code:
        coords = code.circuit.get_detector_coordinates()
    elif type(code) == Cross_Platform_Code:
        coords = code.stim_circuit.get_detector_coordinates()
    else:
        raise TypeError('Code needs to be a Hex_Code or Cross_Platform_Code')
    shots = len(detectors)

    time_max = max([coords[key][2] for key in coords])
    trunc_layers = (0, 1, 2, 3, time_max - 1, time_max)

    num_mmts = {}
    for d in coords:
        if len(coords[d]) == 4:
            key = str([coords[d][0], coords[d][1], coords[d][3]])
        else:
            key = str([coords[d][0], coords[d][1]])

        if coords[d][2] not in trunc_layers:
            # Need to get num_mmts for trunc vs not trunc
            if key in num_mmts:
                num_mmts[key] += 1
            else:
                num_mmts[key] = 1

    # Gathering e/m heatmaps
    layout = code.layout
    max_heat = 1
    e_heatmap, m_heatmap = {}, {}

    for s in range(shots):
        for d in coords:
            # Only if detector triggered
            if detectors[s][d] and coords[d][2] not in trunc_layers:
                if len(coords[d]) == 4:
                    key = str([coords[d][0], coords[d][1], coords[d][3]])
                else:
                    key = str([coords[d][0], coords[d][1]])
                index = index_from_coord(coords[d][0], coords[d][1], layout.width)
                m_type = coords[d][3] == 1 if len(coords[d]) == 4 else (coords[d][0] + coords[d][1]) % 4 // 2
                
                # Make sure its an actual plaquette instead of a link
                if index in layout.plaq_dict:
                    relevent_heatmap = m_heatmap if m_type else e_heatmap
                    if key in relevent_heatmap:
                        relevent_heatmap[key] += 1
                        if relevent_heatmap[key] > max_heat:
                            max_heat = relevent_heatmap[key]
                    else:
                        relevent_heatmap[key] = 1
                elif index in layout.truncated_plaquettes and include_spatial_truncs:
                    relevent_heatmap = m_heatmap if m_type else e_heatmap
                    if key in relevent_heatmap:
                        relevent_heatmap[key] += 1
                        if relevent_heatmap[key] > max_heat:
                            max_heat = relevent_heatmap[key]
                    else:
                        relevent_heatmap[key] = 1


    # Normalizing
    for entry in e_heatmap:
        e_heatmap[entry] = e_heatmap[entry] / (num_mmts[entry] * shots)
    for entry in m_heatmap:
        m_heatmap[entry] = m_heatmap[entry] / (num_mmts[entry] * shots)

    return e_heatmap, m_heatmap
# End get_em_heatmaps


