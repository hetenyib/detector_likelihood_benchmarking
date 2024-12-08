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
from src.layout_planar_main import coord_from_index, index_from_coord, Planar_Layout
import src.layout_direct_borders as direct

def basic_box(width, 
              length,
              noise=1e-3*np.ones(5), 
              noise_dict=None,
              color_equals_pauli=True,
              pauli_cycle=0,
              border=None,
              buffer=2) -> Planar_Layout:
    '''
    Creates a rectangular brick wall similar to the IBM layouts.
    '''
    assert buffer % 2 == 0, 'Use an even number for the buffer.'
    layout_width = 4 * width + 3 + 2 * buffer
    plaq_indexes = [index_from_coord(4 * w + 2 + 2 * (l % 2) + buffer,
                                     2 * l + 1 + buffer,
                                     layout_width)
                        for w in range(width) 
                        for l in range(length)]

    layout = Planar_Layout(plaq_indexes, 
                         layout_width, 
                         noise=noise, 
                         color_equals_pauli=color_equals_pauli,
                         pauli_cycle=pauli_cycle,
                         noise_dict=noise_dict)
    
    layout.template = 'basicbox'
    
    # Creating Border
    #add_truncated_links(layout)
    #add_kesselring_border_plaquettes(layout)
    #layout.border = 'kesselring'

    # Defining observable Location
    #set_observable(layout, 'default', shift=obs_shift) 

    return layout
# End basic_box


def parallelogram(width, 
              length,
              noise=1e-3*np.ones(5), 
              noise_dict=None,
              color_equals_pauli=True,
              pauli_cycle=2,
              buffer=2,
              basis='primary',
              border='direct',
              obs_shift=0) -> Planar_Layout:
    '''
    Create parallelogram shape for Hexagons common for planar layouts.
    '''
    assert buffer % 2 == 0, 'Use an even number for the buffer.'
    layout_width = 4 * (width + length // 2) + 2 * buffer
    plaq_indexes = [index_from_coord(4 * (w + l // 2) + 2 + 2 * (l % 2) + buffer,
                                     2 * l + 1 + buffer,
                                     layout_width)
                        for w in range(width) 
                        for l in range(length)]
    
    layout = Planar_Layout(plaq_indexes, 
                         layout_width, 
                         noise=noise, 
                         color_equals_pauli=color_equals_pauli,
                         pauli_cycle=pauli_cycle,
                         noise_dict=noise_dict)
    layout.template = 'parallelogram'
    
    # Creating Border
    layout.border = border
    if border == 'direct':
        direct.add_truncated_links(layout)
        direct.add_kesselring_border_plaquettes(layout)

    # Defining observable Location
    if border == 'direct':
        direct.set_observable(layout, basis, obs_shift=obs_shift)

    return layout
# End parallelogram


def diamond(size,
            border='direct',
            basis='primary',
            noise=1e-3*np.ones(5),
            noise_dict=None):
    '''
    Defines a diamond layout that will fit on an eagle device.
    Intended to be used with direct truncated plaquettes.
    '''
    assert type(border) == str, f'Border argument given was {border}, not "direct" or None.'
    ### Constructing Layout
    # Defining first plaq and pauli_cycle
    init_x = ((2 * size + 4) // 4) * 4
    width = 2 * init_x

    x0, y0 = init_x, 3
    first_plaq = index_from_coord(x0, y0, width)
    plaq_indexes = [first_plaq]
    pauli_cycle = (((size - 1) // 2) - (size % 2)) % 3

    # Extending one plaquette to full side
    for s in range(1,size):
        plaq_indexes.append(index_from_coord(x0 - 2 * s, y0 + 2 * s, width))

    # Extending side to diamond
    for s in range(size):
        xi, yi = coord_from_index(plaq_indexes[s], width)
        for l in range(1, size):
            plaq_indexes.append(index_from_coord(xi + 2 * l, yi + 2 * l, width))
    
    # Defining the layout
    layout = Planar_Layout(plaq_indexes, 
                           width, 
                           pauli_cycle=pauli_cycle,
                           noise=noise,
                           noise_dict=noise_dict)
    layout.template = 'diamond'

    # Creating border 
    layout.border = border
    if border == 'direct':
        direct.add_truncated_links(layout)
        direct.add_kesselring_border_plaquettes(layout)

    # Defining observable
    if border == 'direct':
        direct.set_observable(layout, basis)
    
    return layout
# End diamond
