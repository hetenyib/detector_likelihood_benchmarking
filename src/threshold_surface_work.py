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
from src.CSS_direct import Hex_Code
from src.layout_templates import diamond
from src.simulation import error_rate_sim

# Just saving some nonsense here
# Not intended as very robust or useful code.
# Creates a 'threshold surface' in a quite
# handwavy and approximate way. 

def varying_error_rate_sim(sub_rounds=18, 
                  noise_breakdown=1*np.ones(5), 
                  basis='primary',
                  first_noises=[1e-3, 6e-3, 1e-2, 5e-2],
                  first_shots=int(1e4),
                  second_shots=int(1e5),
                  printing=False,
                  plotting=False,
                  window=1.5,
                  second_slices=4,
                  variable_sub_rounds=False):
    '''
    Does a low shot scan over large noise region to find neighborhood of threshold, 
    then does a closer pass with higher number of shots to get 
    closest to actual intercept.
    '''
    if printing:
        print('Noise breakdown:', noise_breakdown, end='')
    # First Pass
    first_error_rates_5, first_error_rates_8 = [], []
    for n in first_noises:
        noise = n * noise_breakdown
        
        # Size 5
        if variable_sub_rounds:
            sub_rounds = 2 * 8 # 2*distance
        layout_5 = diamond(5, basis=basis, noise=noise)
        code_5 = Hex_Code(layout_5, sub_rounds=sub_rounds)
        first_error_rates_5.append(error_rate_sim(code_5, first_shots))
        # Size 8
        if variable_sub_rounds:
            sub_rounds = 2 * 12 # 2*distance
        layout_8 = diamond(8, basis=basis, noise=noise)
        code_8 = Hex_Code(layout_8, sub_rounds=sub_rounds)
        first_error_rates_8.append(error_rate_sim(code_8, first_shots))
        if printing:
            print('.', end='')

    # Finding Second Pass range
    first_threshold = get_threshold(first_noises, first_error_rates_8, first_error_rates_5)
    if plotting:
        plot_slop_thresh(first_noises,
                         first_error_rates_5,
                         first_error_rates_8, 
                         first_threshold)
    if printing:
        print('first_thresh:', first_threshold, end=';')
    new_noises = np.linspace(first_threshold / window, first_threshold * window, second_slices)

    # Second Pass
    error_rates_5, error_rates_8 = [], []
    for n in new_noises:
        noise = n * noise_breakdown
        # Size 5
        if variable_sub_rounds:
            sub_rounds = 2 * 8 # 2*distance
        layout_5 = diamond(5, basis=basis, noise=noise)
        code_5 = Hex_Code(layout_5, sub_rounds=sub_rounds)
        error_rates_5.append(error_rate_sim(code_5, second_shots))
        # Size 8
        if variable_sub_rounds:
            sub_rounds = 2 * 12 # 2*distance
        layout_8 = diamond(8, basis=basis, noise=noise)
        code_8 = Hex_Code(layout_8, sub_rounds=sub_rounds)
        error_rates_8.append(error_rate_sim(code_8, second_shots))
        if printing:
            print('.', end='')


    threshold = get_threshold(new_noises, error_rates_8, error_rates_5)
    if printing:
        print(f'threshold:{threshold}')

    if plotting:
        plot_slop_thresh(new_noises,
                         error_rates_5,
                         error_rates_8, 
                         threshold)

    return threshold, error_rates_5, error_rates_8
# End threshold_sim


def below_threshold(sub_rounds=18, 
                  basis='primary',
                  noises=[1.05e-2, 2e-3, 1e-2, 1e-3, 7e-3], # sher noises default
                  shots=int(1e4)):
    
    # Distance 8
    layout_5 = diamond(5, basis=basis, noise=noises)
    code_5 = Hex_Code(layout_5, sub_rounds=sub_rounds)
    error_rate_5 = error_rate_sim(code_5, shots)
    # Distance 12
    layout_8 = diamond(8, basis=basis, noise=noises)
    code_8 = Hex_Code(layout_8, sub_rounds=sub_rounds)
    error_rate_8 = error_rate_sim(code_8, shots)

    return error_rate_8 < error_rate_5, error_rate_8, error_rate_5
# End below_threshold


def get_threshold(noises, large_d, small_d):
    # Look at the sign of the difference
    diff = np.array(small_d) - np.array(large_d)
    sign = np.sign(diff)
    switch = -1
    
    for i in range(len(sign)-1, 0, -1):
        # Ignoring floor noise (zero logical ER), find when sign switches
        # And pick the smallest value (to avoid coin flip switching) by working backwards
        if sign[i] != 0 and sign[i - 1] != 0:
            if sign[i-1] != sign[i]:
                switch = i

    # Set up scan between points
    points = np.linspace(noises[switch-1], noises[switch], 1000)
    interp_diff = np.interp(points, 
                        [noises[switch-1], noises[switch]], 
                        [diff[switch-1], diff[switch]])
    interp_diff_signs = np.sign(interp_diff)

    # Looking for intersection
    threshold = -1
    for i in range(len(interp_diff_signs)):
        if interp_diff_signs[i] != interp_diff_signs[i-1]:
            threshold = points[i]
            flip = i

    debugging = False
    if debugging:
        print(points[flip], points[flip-1])
        print(flip)
        print(interp_diff_signs[flip], interp_diff_signs[flip-1])

        plt.plot([noises[switch-1], noises[switch]], [small_d[switch-1], small_d[switch]],'--', label='small')
        plt.plot([noises[switch-1], noises[switch]], [large_d[switch-1], large_d[switch]],'--', label='large')
        #plt.plot([noises[switch-1], noises[switch]], [diff[switch-1], diff[switch]],'--', label='diff')
        plt.plot(points, abs(interp_diff))
        #plt.plot(points, interp_diff_signs/10, '.')
        #plt.plot(points, np.zeros(len(points)), '--', color='lightgrey')
        plt.legend()
        plt.grid(which='both')
        plt.loglog()
        plt.show()
    
    return threshold
# End sloppy_threshold


def vary_idle_sim(sub_rounds=18, 
                  roi=1e-2,
                  two_gate=7e-3, 
                  basis='primary',
                  noises=[5e-3, 1e-3, 5e-3, 1e-2, 3e-2],
                  first_shots=int(1e4),
                  second_shots=int(1e5),
                  printing=False):
    '''
    Runs broad threshold with parallelogram width = 5 & 8 (d=8, 12) 
    and finds where they intercept.
    Then uses range where intercept is detected to run second sim 
    to determine a more accurate value. 
    Issue arises when intercept is next to a first pass point
    then due to noise lands outside of second pass. (returns -1 aka no intercept.)
    '''
    first_error_rates_5, first_error_rates_8 = [], []
    # First Pass
    for n in noises:
        noise = [roi, n, roi, 1e-4, two_gate]
        # Distance 8
        layout_5 = diamond(5, basis=basis, noise=noise)
        code_5 = Hex_Code(layout_5, sub_rounds=sub_rounds)
        first_error_rates_5.append(error_rate_sim(code_5, first_shots))
        # Distance 12
        layout_8 = diamond(8, basis=basis, noise=noise)
        code_8 = Hex_Code(layout_8, sub_rounds=sub_rounds)
        first_error_rates_8.append(error_rate_sim(code_8, first_shots))
        if printing:
            print('.', end='')

    # Finding Second Pass range
    first_thresh = get_threshold(noises, first_error_rates_8, first_error_rates_5)
    if printing:
        print('first_thresh:', first_thresh)
    new_noises = np.linspace(first_thresh * .5, first_thresh * 1.75, 5)
    if printing:
        print(f'New noises: {new_noises}')

    # Second Pass
    error_rates_5, error_rates_8 = [], []
    for n in new_noises:
        noise = [roi, n, roi, 1e-4, two_gate]
        # Distance 8
        layout_5 = diamond(5, basis=basis, noise=noise)
        code_5 = Hex_Code(layout_5, sub_rounds=sub_rounds)
        error_rates_5.append(error_rate_sim(code_5, second_shots))
        # Distance 12
        layout_8 = diamond(8, basis=basis, noise=noise)
        code_8 = Hex_Code(layout_8, sub_rounds=sub_rounds)
        error_rates_8.append(error_rate_sim(code_8, second_shots))
        if printing:
            print('.', end='')

    if printing:
        print(f'5:{error_rates_5}; 8:{error_rates_8}')

    return round(get_threshold(new_noises, error_rates_5, error_rates_8), 6)
# End vary_idle_sim

def plot_slop_thresh(n, er5, er8, thres, title_add=''):
    plt.plot(n, er5, 'o--', label='d=8')
    plt.plot(n, er8, 'o--', label='d=12')
    plt.vlines(thres, min(er8), .5, linestyles='dashed', color='lightgray')
    plt.legend()
    plt.loglog()
    plt.title(f'Threshold: {thres}' + title_add)
    plt.grid(which='both')
    plt.show()
# End plot_slop_thresh


def get_intercepts(shots=int(1e5)):
    # [ROI, idle, ROI, 1gate, 2gate]
    breakdown_ROI = np.array([1e-2, 0, 1e-2, 1e-4, 0]) / 1e-2
    breakdown_idle = np.array([0, 2e-3, 0, 1e-4, 0]) /  2e-3
    breakdown_two_gate = np.array([0, 0, 0, 1e-4, 7e-3]) / 7e-3

    # ROI
    print('ROI', end='')
    noises = np.linspace(2.5, 3.25, 5) * 1e-2
    er5_ROI, er8_ROI = varying_error_rate_sim(noises=noises, noise_breakdown=breakdown_ROI, shots=shots)
    print(',')
    thresh_ROI = get_threshold(noises, er8_ROI, er5_ROI)
    plot_slop_thresh(noises, er5_ROI, er8_ROI, thresh_ROI)

    # idle
    print('idle', end='')
    noises = np.linspace(1.5, 2.5, 5) * 1e-2
    er5_idle, er8_idle = varying_error_rate_sim(noises=noises, noise_breakdown=breakdown_idle, shots=shots)
    print(',')
    thresh_idle = get_threshold(noises, er8_idle, er5_idle)
    plot_slop_thresh(noises, er5_idle, er8_idle, thresh_idle)

    # 2gate
    print('2gate', end='')
    noises = np.linspace(1.5, 2, 5) * 1e-2
    er5_two_gate, er8_two_gate = varying_error_rate_sim(noises=noises, noise_breakdown=breakdown_two_gate, shots=shots)
    print(',')
    thresh_two_gate = get_threshold(noises, er8_two_gate, er5_two_gate)
    plot_slop_thresh(noises, er5_two_gate, er8_two_gate, thresh_two_gate)

    print('ROI threshold:',thresh_ROI)
    print('idle threshold:',thresh_idle)
    print('2gate threshold:',thresh_two_gate)

    return thresh_ROI, thresh_two_gate, thresh_idle

    # Default shots ~ 13-14s for one of them.
# End get_intercepts


def sim_and_plot_surface(thresh_ROI, thresh_two_gate, thresh_idle,
                         granularity=11,
                         shots=int(2**15)):
    '''
    Take intercepts directly from get_intercepts.
    Warning: can take like an hour.
    Adjust granularity for density of points. 
    '''
    # Creating Surface
    x_int, y_int, z_int = thresh_ROI, thresh_two_gate, thresh_idle
    a = -z_int / x_int
    b = -z_int / y_int
    c = z_int

    x = np.linspace(0, x_int, granularity)
    y = np.linspace(0, y_int, granularity)
    x, y = np.meshgrid(x, y)
    z_flat = a * x + b * y + c
    z = -1 * np.ones((granularity, granularity))
    for x_loc in range(granularity):
        for y_loc in range(granularity):
            if x_loc + y_loc < granularity:
                z[x_loc][y_loc] = vary_idle_sim(roi=x[x_loc][y_loc], two_gate=y[x_loc][y_loc], shots=shots)
                print(f'{z[x_loc][y_loc]},', end='')
        print('!')
    masked_z = np.ma.masked_where(z < 0, z)

    masked_z_flat = np.ma.masked_where(z_flat < 0, z_flat)
    # Plotting surface
    # x: ROI, y: 2gate, z: idle
    fig = plt.figure(figsize=(8,12))
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(x, y, masked_z, color='tab:blue', alpha=.5)
    ax.plot_surface(x, y, masked_z_flat, color='tab:green', alpha=.5)

    # Device Error Rate Location
    sher_x, sher_y, sher_z = 1e-2, 7e-3, 2e-3
    plt.plot([sher_x], [sher_y], [sher_z], marker='o', color='tab:red', zorder=3)
    plt.plot([sher_x, 0], [sher_y, sher_y], [sher_z, sher_z], linestyle='dotted', color='tab:orange', alpha=1)
    plt.plot([sher_x, sher_x], [sher_y, 0], [sher_z, sher_z], linestyle='dotted', color='tab:orange', alpha=1)
    plt.plot([sher_x, sher_x], [sher_y, sher_y], [sher_z, 0], linestyle='dotted', color='tab:orange', alpha=1)

    # Optional log scalling 
    log_scale = False
    if log_scale:
        min_p = 1e-4
        ax.set_zlim(min_p, 1.2*z_int)
        ax.set_xlim(min_p, 1.2*x_int)
        ax.set_ylim(min_p, 1.2*y_int)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_zscale('log')

    # Niceties
    ax.set_xlabel('ROI')
    ax.set_ylabel('2gate')
    ax.set_zlabel('idle')
    ax.view_init(5,80)
    plt.show()
# End sim_and_plot_surface
