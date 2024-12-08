# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from src.CSS_direct import Hex_Code
from src.qiskit_glue import Cross_Platform_Code
from src.effective_p import log_er_per_round


def plot_shot(code:Hex_Code or Cross_Platform_Code, 
              syndromes=None, 
              pairs=[], 
              figsize=(8,16), 
              scaling=(1,1,1), 
              title='Syndrome Graph', 
              view=(85,-5),
              axis=False,
              edge_heatmap={},
              cmap_key='seismic',
              highlights=[],
              cmap_default = .5,
              circular_shading=False):
    '''
    Plots a visualization of the decoding graph with edge connections.
    takes arguments syndromes and pairs to visualize a single shot of noise and/or decoding results
    '''
    # Basics
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    colors = ['tab:red', 'tab:blue', 'tab:green', 'red', 'blue', 'green', 'tab:purple']

    if type(code) == Cross_Platform_Code:
        detector_coords = code.stim_circuit.get_detector_coordinates()
    else:
        detector_coords = code.circuit.get_detector_coordinates()
    if syndromes is None:
        syndromes = np.zeros(len(detector_coords))
    
    # Plotting detectors with syndromes
    for coord in detector_coords:
        if len(detector_coords[coord]) == 4:
            x, y, t, c = detector_coords[coord]
        elif len(detector_coords[coord]) == 3:
            x, y, t = detector_coords[coord]
            c =  (x + y) % 4 // 2
        # Syndrome outline
        if syndromes[coord]:
            ax.scatter(x, y, t, color='red', alpha=.5, s=100, zorder=1)
        # Detector dot
        color = 'fuchsia' if coord in highlights else colors[int(c)]
        ax.scatter(x, y, t, color=color, s=50, zorder=2)

    # Plotting edges
    cmap = mpl.colormaps[cmap_key]
    no_heatmap = len(edge_heatmap) == 0
    for edge in code.match.edges():
        node0, node1 = edge[0], edge[1]
        # Boundary Edge
        if node1 is None:
            e = np.array([node0, -1])
            if no_heatmap:
                shade = 'lightgray'
            elif str(e) in edge_heatmap:
                if circular_shading:
                    shade = cmap(np.sqrt(1 - (edge_heatmap[str(e)] - 1)**2))
                else:
                    shade = cmap(edge_heatmap[str(e)])
            else:
                shade = cmap(cmap_default)
            coord_0 = detector_coords[node0]
            plt.plot([coord_0[0], coord_0[0] + .4], 
                        [coord_0[1], coord_0[1] + .6],
                        [coord_0[2], coord_0[2] + .5],
                        c=shade, zorder=-2, linewidth=2)
        # Normal edge between two points
        else:
            e = np.array([node0, node1])
            if no_heatmap:
                shade = 'lightgray'
            elif str(e) in edge_heatmap:
                if circular_shading:
                    shade = cmap(np.sqrt(1 - (edge_heatmap[str(e)] - 1)**2))
                else:
                    shade = cmap(edge_heatmap[str(e)])
            else:
                shade = cmap(cmap_default)
            coord_0 = detector_coords[node0]
            coord_1 = detector_coords[node1]
            plt.plot([coord_0[0], coord_1[0]], 
                        [coord_0[1], coord_1[1]],
                        [coord_0[2], coord_1[2]],
                        c=shade, zorder=-2, linewidth=.5)
            
    # Plotting matchings
    for pair in pairs:
        if pair[1] >= 0:
            coord_0 = detector_coords[pair[0]]
            coord_1 = detector_coords[pair[1]]
            plt.plot([coord_0[0], coord_1[0]], 
                     [coord_0[1], coord_1[1]],
                     [coord_0[2], coord_1[2]],
                     color='red', zorder=-1)
        else:
            coord_0 = detector_coords[pair[0]]
            plt.plot([coord_0[0], coord_0[0] + .4], 
                     [coord_0[1], coord_0[1] + .6],
                     [coord_0[2], coord_0[2] + .5],
                     color='red', zorder=-1, linewidth=2)

    # Polishing appearance
    plt.title(title)
    ax.set_xlabel('Width')
    ax.set_ylabel('Length')
    ax.set_zlabel('Time')
    ax.set_box_aspect(aspect = scaling) 
    ax.view_init(view[0], view[1])
    if not axis:
        ax.set_axis_off()

    plt.show()
# End plot_shot


def plot_simulation_comparison(real_data, sim_fits,
                               title='Simulation Comparison',
                               fig_size=(4,6)):
    '''
    real_data is the dictionary that rp.get_error_rate_per_round_data outputs
    sim_data is the dictionary that rp.get_simulation_comparison outputs
    
    '''
    plt.figure(figsize=fig_size)
    labels = ['Average', 'Calibrated', 'Effective-p']
    c = 0
    for label in labels:
        plt.plot([key for key in sim_fits[label]['data']], [sim_fits[label]['data'][key] for key in sim_fits[label]['data']], 
                'o', label=label, color=f'C{c}')
        plt.errorbar([key for key in sim_fits[label]['data']], [sim_fits[label]['data'][key] for key in sim_fits[label]['data']], 
                    yerr=1/np.sqrt(sim_fits[label]['shots']), linestyle=' ', color=f'C{c}')
        # Fitting
        max_sub_rounds = max(real_data['data']['sub_rounds'])
        plt.plot(np.linspace(4, max_sub_rounds, 1000), 
                log_er_per_round(np.linspace(4, max_sub_rounds, 1000), sim_fits[label]['fit']), 
                '--', color=f'C{c}', label=f'fit: {round(sim_fits[label]["fit"], 4)}  +- {round(sim_fits[label]["error"], 4)}')
        c += 1
        

    # Labels and real data
    fitting = True
    state = {'primary': '|0>', 'dual': '|+>'}
    dd = {'XZX': 'XZXXZX', 'True': 'XX', 'XXXX': 'XXXX', 'False': 'None'}

    label = 'size ' + str(real_data['meta_data']['size']) 
    label += '-loc ' + str(real_data['meta_data']['location'])
    label += '-DD ' + dd[real_data['meta_data']['DD']]
    label += '-' + state[real_data['meta_data']['basis']]
    plt.plot(real_data['data']['sub_rounds'], real_data['data']['error_rates'], 'o', label=label, color=f'C{c}')
    plt.errorbar(real_data['data']['sub_rounds'], real_data['data']['error_rates'], yerr=1/np.sqrt(real_data['data']['shots']), linestyle=' ', color=f'C{c}')
    
    # Plotting Fits
    if fitting:
        plt.plot(np.linspace(4, max_sub_rounds, 1000), 
                log_er_per_round(np.linspace(4, max_sub_rounds, 1000), real_data['data']['round_error_rate']), 
                '--', color=f'C{c}', 
                # label=f'fit: {round(real_data["data"]["round_error_rate"], 4)} +- {round(real_data["data"]["fit_error"], 4)}'
                label='$P_L$ = ' +str(round(100*real_data["data"]["round_error_rate"],1))+'%'
                )

    # Nicities
    plt.grid()
    plt.xlabel('Sub_rounds')
    plt.ylabel('Logical Error Rate')
    default_title = f'Fitting Logical Error Rate per Sub-round (eff_p={round(sim_fits["eff_p"], 4)})'
    plt.title(default_title if title is None else title + f' (eff_p={round(sim_fits["eff_p"], 4)})')
    # plt.hlines(.5, 4, max_sub_rounds + .5, linestyles='dashed', color='gray', label='pure noise')
    plt.legend(loc='lower right')
    plt.show()
# End plot_simulation_comparison

