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
from src.measurement_tracker import *
from src.layout_planar_main import Planar_Layout


def get_plaquette_detectors(sub_round, 
                            layout:Planar_Layout, 
                            mtrack:measurement_tracker,
                            partial=False) -> stim.Circuit:
    '''
    Creates Plaquettes for time-wise bulk of the code.
    '''
    detectors = stim.Circuit()
    color_type = (sub_round + 1) % 3
    pauli_type = 0 if sub_round % 2 == 1 else 1


    all_plaquettes = {**layout.plaq_dict, **layout.truncated_plaquettes}
    for p in all_plaquettes:
        plaq = all_plaquettes[p]
        if plaq.color_type == color_type:
            coord = layout.coord_from_index(plaq.index)
            plaq_links = [link.index for link in plaq.all_links]

            targets = []
            targets += mtrack.get_mmt_targets(plaq_links, -1)
            if not partial:
                targets += mtrack.get_mmt_targets(plaq_links, -5)

            # Makes sure to not create incorrect anyon type detector at boundaries
            if (len(targets) >= 4
             or (partial and len(targets) >= 2)):
                detectors.append_operation('DETECTOR',
                                    [stim.target_rec(target) 
                                        for target in targets],
                                        (coord[0], coord[1], 0, pauli_type))

    return detectors
# End get_plaquette_detectors


def get_partial_init_plaq_detectors(sub_round, 
                            layout:Planar_Layout, 
                            mtrack:measurement_tracker) -> stim.Circuit:
    '''
    Because the partial detectors need to avoid anti-commuting, 
    need to define plaquettes around color and color - 1 (instead of color + 1)
    because those plaquettes share links with the color that was just measured
    in the opposite basis and will commute (hopefully).
    NOTE: Is this used at all?
    '''
    detectors = stim.Circuit()
    #color_type = sub_round % 3
    pauli_type = 0 if sub_round % 2 == 1 else 1
    #border_plaquettes = [p for p in layout.get_border_plaquettes()]

    for p in layout.plaq_dict:
        plaq = layout.plaq_dict[p]
        coord = layout.coord_from_index(plaq.index)
        plaq_links = [link.index for link in plaq.all_links]

        targets = []
        targets += mtrack.get_mmt_targets(plaq_links, -1)
        targets += mtrack.get_mmt_targets(plaq_links, -5)

        # Ensures only taking correct sided plaquettes for truncated cases
        if len(targets) >= 4 :
            detectors.append_operation('DETECTOR',
                                   [stim.target_rec(target) 
                                    for target in targets],
                                    (coord[0], coord[1], 0, pauli_type))

    return detectors
# End get_partial_init_plaq_detectors



def get_init_check_detectors(layout:Planar_Layout, 
                            mtrack:measurement_tracker):
    '''
    Check detectors for the first sub_round in the code,
    Just checking that each of the first check operators are zero.
    Need to select in initialization a first round that matches init logical op type.
    '''
    detectors = stim.Circuit()
    mmts = mtrack.get_previous_round(-1).mmts
    for m in mmts:
        x, y = layout.coord_from_index(m)
        targets = mtrack.get_mmt_targets([m], -1)
        detectors.append_operation('DETECTOR',
                                   [stim.target_rec(target)
                                    for target in targets],
                                    (x, y, 0, ['primary', 'dual'].index(layout.basis)))
    return detectors
# End get_check_detectors


def FTRO_detectors(layout:Planar_Layout, 
                    mtrack:measurement_tracker,
                    sub_round:int) -> stim.Circuit:
    '''
    
    '''
    detectors = stim.Circuit()
    # Ends with Z type, need check operators
    if ((sub_round % 2 == 0 and layout.basis == 'primary')
     or (sub_round % 2 == 1 and layout.basis == 'dual')):
        detectors += get_half_plaq_detectors(layout, mtrack, sub_round)
        detectors.append_operation('SHIFT_COORDS', (),(0,0,1))
        detectors += get_RO_check_detectors(layout, mtrack)
    # Ends with X type, no check operators
    else:
        detectors += get_half_plaq_detectors(layout, mtrack, sub_round)
        #detectors += get_RO_check_detectors(layout, mtrack, sub_round)
        pass
    return detectors
# End FTRO_detectors


def get_RO_check_detectors(layout:Planar_Layout, 
                            mtrack:measurement_tracker):
    '''
    Check detectors for the first sub_round in the code,
    Just checking that each of the first check operators are zero.
    Need to select in initialization a first round that matches init logical op type.
    '''
    detectors = stim.Circuit()
    link_indexes = mtrack.get_previous_round(-2).mmts
    for index in link_indexes:
        x, y = layout.coord_from_index(index)
        targets = mtrack.get_mmt_targets([index], -2)
        targets += mtrack.get_mmt_targets(get_adj_data_qubits(layout, index), -1)
        
        detectors.append_operation('DETECTOR',
                                   [stim.target_rec(target)
                                    for target in targets],
                                    (x, y, 0, ['primary', 'dual'].index(layout.basis)))
    return detectors
# End get_check_detectors


def get_adj_data_qubits(layout:Planar_Layout,
                        index:int):
    '''
    Takes in link index and returns adjacent data qubits.
    Needs to take into account truncated or non-truncated.
    '''
    adjs = []
    if index in layout.link_dict:
        link = layout.link_dict[index]
        adjs.append(link.adj0)
        adjs.append(link.adj1)

    elif index in layout.truncated_links:
        link = layout.truncated_links[index]
        if link.adj0 is not None:
            adjs.append(link.adj0)
        if link.adj1 is not None:
            adjs.append(link.adj1)
    return adjs
# End get_adj_data_qubits


def get_half_plaq_detectors(layout:Planar_Layout, 
                            mtrack:measurement_tracker,
                            sub_round:int):
    '''
    These detectors create a full plaquette with the RO mmts
    then compares them to the previous plaquette formation.
    '''
    detectors = stim.Circuit()
    color = sub_round % 3
    # Even: need just -3 (-ro) half plaqs (Same color as 'color')
    if ((sub_round % 2 == 0 and layout.basis == 'primary')
     or (sub_round % 2 == 1 and layout.basis == 'dual')):
        plaq_color = (color - 1) % 3
        # Bulk Plaquettes
        all_plaquettes = {**layout.plaq_dict, **layout.truncated_plaquettes}
        for p in all_plaquettes:
            plaq = all_plaquettes[p]
            if plaq.color_type == plaq_color:
                plaq_links = [link.index for link in plaq.all_links]
                targets = mtrack.get_mmt_targets(plaq_links, -3 - 1)
                targets += mtrack.get_mmt_targets(plaq.data_qubits, -1)
                coords = (plaq.x, plaq.y, 0, ['primary', 'dual'].index(layout.basis))
                
                if len(targets) > 4:
                    detectors.append_operation('DETECTOR',
                                                [stim.target_rec(target) 
                                                for target in targets],
                                                coords)
    # Odd: need -2(-ro) (color + 1) & -4(-ro) (color - 1) half plaqs
    else:
        for round in [-1, +1]:
            plaq_color = (color - 1 + round) % 3
            # Bulk Plaquettes
            all_plaquettes = {**layout.plaq_dict, **layout.truncated_plaquettes}
            for p in all_plaquettes:
                plaq = all_plaquettes[p]
                if plaq.color_type == plaq_color:
                    plaq_links = [link.index for link in plaq.all_links]
                    targets = mtrack.get_mmt_targets(plaq_links, -3 + round - 1)
                    targets += mtrack.get_mmt_targets(plaq.data_qubits, -1)
                    coords = (plaq.x, plaq.y, 0, ['primary', 'dual'].index(layout.basis))
                    if len(targets) > 4:
                        detectors.append_operation('DETECTOR',
                                                    [stim.target_rec(target) 
                                                    for target in targets],
                                                    coords)
            
            detectors.append_operation('SHIFT_COORDS', (),(0,0,1))
    return detectors
# End get_half_plaq_detectors
