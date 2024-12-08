# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from src.layout_planar_main import *


def get_border_plaquettes(layout:Planar_Layout):
    '''
    Returns non-truncated plaquettes that are exposed
    to a boundary through at least a full edge by checking
    if it has less than 6 surrounding plaquettes (including truncated ones).
    '''
    border_plaqs = {}
    for p in layout.plaq_dict:
        plaq = layout.plaq_dict[p]
        # Exposed to boundary
        if len(get_adjacent_plaquettes(layout, p)) < 6 and not plaq.truncated:
            border_plaqs[plaq.index] = plaq 
    return border_plaqs
# End get_border_plaquettes

def get_side_truncated_links(layout:Planar_Layout, m_type):
    '''
    Returns correct style of side links.
    '''
    if layout.template == 'parallelogram':
        return get_parallelogram_side_truncated_links(layout, m_type)
    elif layout.template == 'diamond':
        return get_diamond_side_truncated_links(layout, m_type)
    else:
        return []
# End get_side_truncated_links

def get_side_plaquettes(layout:Planar_Layout, 
                        want_m_type:bool, 
                        remove:int=None):
    '''
    Returns correct style of side plaquettes.
    '''
    if layout.template == 'parallelogram':
        return get_parallelogram_side_plaquettes(layout, want_m_type)
    elif layout.template == 'diamond':
        return get_diamond_side_plaquettes(layout, want_m_type)
    else:
        return []
# End get_side_plaquettes


def get_parallelogram_side_truncated_links(layout:Planar_Layout, m_type):
    '''
    Determines which links to be included in a m or e type border.
    '''
    min_data = min(layout.data_qubits)
    max_data = max(layout.data_qubits)
    min_y = coord_from_index(min_data, layout.width)[1]
    max_y = coord_from_index(max_data, layout.width)[1]

    # Defining flat sizes of the parallelogram
    min_y_data_qubits = [index for index in layout.truncated_links
                         if min_y == coord_from_index(index, layout.width)[1]]
    max_y_data_qubits = [index for index in layout.truncated_links
                         if max_y == coord_from_index(index, layout.width)[1]]

    # Chopping ends to make m-type list
    m_links = sorted(min_y_data_qubits)[:-1] + sorted(max_y_data_qubits)[1:]

    # e-type the non-m-types plus the corner qubits.
    e_links = [index for index in layout.truncated_links
              if index not in m_links]
    e_links += [min_data, max_data]

    if m_type:
        return m_links
    else:
        return e_links
# End get_side_truncated_links

def get_parallelogram_side_plaquettes(layout:Planar_Layout, 
                        want_flat_side:bool, 
                        remove:int=None):
    '''
    Returns truncated plaquettes of one side or the other.
    '''
    border_plaqs = get_border_plaquettes(layout)
    border_ys = [border_plaqs[p].y for p in border_plaqs]
    min_y, max_y = min(border_ys), max(border_ys)

    step_side = []
    flat_side = []
    for p in layout.truncated_plaquettes:
        plaq = layout.truncated_plaquettes[p]
        if plaq.color_type != remove:
            if min_y <= plaq.y <= max_y:
                step_side.append(p)
            else:
                flat_side.append(p)

    if want_flat_side:
        return flat_side
    else:
        return step_side
# End get_parallelogram_side_plaquettes


def get_diamond_side_truncated_links(layout:Planar_Layout, m_type):
    '''
    Determines which links to be included in a m or e type border.
    For diamond layout.
    '''
    # Initializing variables
    m_links = []
    e_links = []
    min_x, min_y = coord_from_index(min(layout.data_qubits), 
                                    layout.width)
    min_sum = min_x + min_y
    max_x, max_y = coord_from_index(max(layout.data_qubits), 
                                    layout.width)
    max_sum = max_x + max_y

    # Sorting side links
    for p in layout.truncated_links:
        x, y = coord_from_index(p, layout.width)
        sum = x + y
        if min_sum < sum < max_sum:
            e_links.append(p)
        else:
            m_links.append(p)

    # Adding shared truncated link
    sorted_data_qubits = sorted(layout.data_qubits)
    m_links += [sorted_data_qubits[1], sorted_data_qubits[-2]]

    # Returning desired set of side links.
    if m_type:
        return m_links
    else:
        return e_links
# End get_diamond_side_truncated_links

def get_diamond_side_plaquettes(layout:Planar_Layout, m_type):
    '''
    Determines which links to be included in a m or e type border.
    For diamond layout.
    '''
    # Initializing variables
    m_plaqs = []
    e_plaqs = []
    min_x, min_y = coord_from_index(min(layout.data_qubits), 
                                    layout.width)
    min_sum = min_x + min_y
    max_x, max_y = coord_from_index(max(layout.data_qubits), 
                                    layout.width)
    max_sum = max_x + max_y

    # Sorting side plaquettes
    for p in layout.truncated_plaquettes:
        x, y = coord_from_index(p, layout.width)
        sum = x + y
        if min_sum < sum < max_sum:
            e_plaqs.append(p)
        else:
            m_plaqs.append(p)

    # Returning desired set of side plaquettes.
    if m_type:
        return m_plaqs
    else:
        return e_plaqs
# End get_diamond_side_truncated_links


def add_kesselring_border_plaquettes(layout:Planar_Layout):
    '''
    Adds kesselring style wt-3 border plaquettes along boundary. 
    Assumed truncated links are already added
    '''
    for plaq in layout.plaq_dict:
        neighbor_plaquettes = get_adjacent_plaquettes(layout, plaq)
        # It is a border plaquette
        if len(neighbor_plaquettes) < 6:
            # Check all adjacent plaquettes to find if any are also borders
            for adj_plaq in neighbor_plaquettes:
                adj_plaq_neighbors = get_adjacent_plaquettes(layout, adj_plaq)
                # adj_plaq also a border plaquette
                if len(adj_plaq_neighbors) < 6:
                    shared_neighbors = [index 
                                        for index in get_adjacent_plaquette_locations(layout, plaq)
                                        if index in get_adjacent_plaquette_locations(layout, adj_plaq)]
                    # Check shared_neighbors to see if there are any sites for a truncated plaquette.
                    for shared_plaq in shared_neighbors:
                        if shared_plaq in layout.plaq_dict:
                            pass # Plaquette exists in bulk
                        elif shared_plaq in layout.truncated_plaquettes:
                            pass # Plaquette already has a truncated plaquette
                        else:
                            add_truncated_plaquette(layout, shared_plaq)

    # Determining border plaquette anyon types.    
    m_type_plaquettes = get_side_plaquettes(layout, True)
    e_type_plaquettes = get_side_plaquettes(layout, False)

    # Updating anyon type values
    for p in layout.truncated_plaquettes:
        plaq = layout.truncated_plaquettes[p]
        if plaq.index in m_type_plaquettes:
            plaq.anyon_type = 'm'
        if plaq.index in e_type_plaquettes:
            plaq.anyon_type = 'e'
# End add_kesselring_border_plaquettes

   
def add_truncated_plaquette(layout:Planar_Layout, 
                            index):
    '''
    Adds truncated border plaquette for a given index by 
    checking all of the links (then truncated links) around the index given 
    and adds the final object to the truncated_plaquettes dictionary.
    '''
    adjacent_links = [layout.link_dict[link_index] 
                        for link_index in get_adjacent_link_indexes(index, layout.width)
                        if link_index in layout.link_dict]
    adj_truncated_links = [layout.truncated_links[truncated_link_index]
                            for truncated_link_index in get_adjacent_data_qubit_indexes(index, layout.width)
                            if truncated_link_index in layout.truncated_links]
    
    layout.truncated_plaquettes[index] = Plaquette(index, 
                                                    layout.width, 
                                                    adjacent_links + adj_truncated_links,
                                                    truncated=True)
# End add_truncated_plaquette 


def add_truncated_links(layout:Planar_Layout):
    '''
    Adds truncated links to the planar code around the boundaries. 

    TODO: Most likely add new class or something to be able to capture
    exactly when to measure. Only measure half of the time. Tied to which
    pauli type the border its a part of. Likely can so a simple Y position calculation.
    (Have to be careful of corners though.)
    *ISSUE*: Wrapping around problem!!! basic_box rough edges show this.
    fixed by padding extra space with width.
    '''
    trunc_qubits = {}
    width = layout.width
    for p in layout.plaq_dict:
        # Check every direction to see if there is a link in self.links, if not, add trunc
        # ^ 
        plaq = layout.plaq_dict[p]
        x, y = plaq.x, plaq.y
        # <^
        if index_from_coord(x - 3, y - 1, width) not in layout.link_dict: 
            trunc_link_index = index_from_coord(x - 2, y - 1, width)
            trunc_qubits[trunc_link_index] = Link(trunc_link_index, layout.width, layout.color_equals_pauli, 0, color_type=plaq.color_type, truncated=True)
        # ^>
        if index_from_coord(x - 3, y + 1, width) not in layout.link_dict: 
            trunc_link_index = index_from_coord(x - 2, y + 1, width)
            trunc_qubits[trunc_link_index] = Link(trunc_link_index, layout.width, layout.color_equals_pauli, 0, color_type=plaq.color_type, truncated=True)
        # .>
        if index_from_coord(x + 3, y - 1, width) not in layout.link_dict: 
            trunc_link_index = index_from_coord(x + 2, y - 1, width)
            trunc_qubits[trunc_link_index] = Link(trunc_link_index, layout.width, layout.color_equals_pauli, 0, color_type=plaq.color_type, truncated=True)
        # <.
        if index_from_coord(x + 3, y + 1, width) not in layout.link_dict: 
            trunc_link_index = index_from_coord(x + 2, y + 1, width)
            trunc_qubits[trunc_link_index] = Link(trunc_link_index, layout.width, layout.color_equals_pauli, 0, color_type=plaq.color_type, truncated=True)
        # <
        if index_from_coord(x, y - 2, width) not in layout.link_dict:     
            trunc_link_index = index_from_coord(x, y - 1, width)
            trunc_qubits[trunc_link_index] = Link(trunc_link_index, layout.width, layout.color_equals_pauli, 0, color_type=plaq.color_type, truncated=True)
        # >
        if index_from_coord(x, y + 2, width) not in layout.link_dict:     
            trunc_link_index = index_from_coord(x, y + 1, width)
            trunc_qubits[trunc_link_index] = Link(trunc_link_index, layout.width, layout.color_equals_pauli, 0, color_type=plaq.color_type, truncated=True)
    
    # "Chopping" links (removing adj that is not in the layout.)
    for t in trunc_qubits:
        link = trunc_qubits[t]
        if link.adj0 not in layout.data_qubits:
            link.adj0 = None
        if link.adj1 not in layout.data_qubits:
            link.adj1 = None
        link.adj0 = link.index
    layout.truncated_links = trunc_qubits
    #self.link_dict.update(self.truncated_links)
# End add_truncated_links


def set_observable(layout:Planar_Layout,
                   obs_style:str,
                   obs_shift:int=0):
    '''
    Sets observable for given layout based on template type. 
    Only compatable with defined layout types. 
    '''
    if layout.template == 'parallelogram':
        set_parallelogram_observable(layout, obs_style, obs_shift)
    elif layout.template == 'diamond':
        set_diamond_observable(layout, obs_style, obs_shift)
    else:
        print(f'No observable set because the {layout.template} template does not have defined logical observables yet.')
# End set_observable


def set_parallelogram_observable(layout:Planar_Layout, 
                   obs_style, 
                   obs_shift=0):
    '''
    Defines locations of the observable, including data qubit
    locations as well as which link operators to multiply in later.
    '''
    if obs_style == 'primary':
        # Getting correct column start and end points
        first_plaq_index = min([d for d in layout.plaq_dict
                                if not layout.plaq_dict[d].truncated])
        qubit_in_column = max(layout.plaq_dict[first_plaq_index].data_qubits)
        column_start = layout.width * ((qubit_in_column // layout.width) + 2*obs_shift)
        column_end = column_start + layout.width

        # Collecting links
        obs_links = [index 
                        for index in layout.link_dict
                        if column_start <= index < column_end]
        obs_links += [index
                        for index in layout.truncated_links
                        if column_start <= index < column_end]
        
        # Collecting data qubits
        obs_data_qubits = [index 
                        for index in layout.data_qubits
                        if column_start <= index < column_end]
        
        # Update layout basis type
        layout.basis = 'primary'
        
    elif obs_style == 'dual':
        # defining variables to find observable location
        first_plaq_index = min([d for d in layout.plaq_dict
                                if not layout.plaq_dict[d].truncated])
        start_qubit = min(layout.plaq_dict[first_plaq_index].data_qubits) + 2
        start_x, start_y = layout.coord_from_index(start_qubit)
        min_diff = start_x - start_y
        max_diff = start_x - start_y + 2

        # Collecting links
        obs_links =  [index 
                        for index in layout.link_dict
                        if min_diff <= 
                        layout.link_dict[index].x - layout.link_dict[index].y
                        <= max_diff]
        obs_links += [index 
                        for index in layout.truncated_links
                        if min_diff <= 
                        layout.truncated_links[index].x - layout.truncated_links[index].y
                        <= max_diff]
        
        # Collecitng data qubits
        obs_data_qubits =  [index 
                            for index in layout.data_qubits
                            if min_diff <= 
                            layout.coord_from_index(index)[0] - layout.coord_from_index(index)[1]   
                            <= max_diff]
        
        # Updating layout basis type
        layout.basis = 'dual'
    else:
        raise Exception('Select "primary" or "dual" basis.')

    # Adding links and data qubits to layout object
    layout.observable_links = obs_links
    layout.observable_data_qubits = obs_data_qubits
# End get_parallelogram_observable


def set_diamond_observable(layout:Planar_Layout,
                           obs_style:str,
                           obs_shift:int=0):
    '''
    Defining observable for the diamond layout.
    '''
    min_x, min_y = coord_from_index(min(layout.data_qubits), layout.width)
    if obs_style == 'primary':
        # Getting correct column start and end points
        x0, y0 = min_x + 4, min_y
        coord_sum = x0 + y0

        

        # Collecting links
        obs_links = [index
                            for index in layout.link_dict
                            if coord_sum <=
                            layout.link_dict[index].x + layout.link_dict[index].y
                            <= coord_sum + 2]
        obs_links += [index
                            for index in layout.truncated_links
                            if coord_sum <=
                            layout.truncated_links[index].x + layout.truncated_links[index].y
                            <= coord_sum + 2]
        
        # Collecting data qubits
        obs_data_qubits = [index
                            for index in layout.data_qubits
                            if coord_sum <=
                            layout.coord_from_index(index)[0] + layout.coord_from_index(index)[1]
                            <= coord_sum + 2]
        # Need to adjust order of data qubits for readout ordering
        for i in np.arange(1, len(obs_data_qubits) - 2, 2):
            obs_data_qubits[i], obs_data_qubits[i+1] = obs_data_qubits[i+1], obs_data_qubits[i]
        
        # Update layout basis type
        layout.basis = 'primary'
        
    elif obs_style == 'dual':
        # defining variables to find observable location
        coord_diff = min_x - min_y

        # Collecting links
        obs_links =  [index
                        for index in layout.link_dict
                        if coord_diff - 2 <=
                        layout.link_dict[index].x - layout.link_dict[index].y
                        <= coord_diff]
        obs_links += [index
                        for index in layout.truncated_links
                        if coord_diff - 2 <=
                        layout.truncated_links[index].x - layout.truncated_links[index].y
                        <= coord_diff]
        
        # Collecitng data qubits
        obs_data_qubits =  [index
                            for index in layout.data_qubits
                            if coord_diff - 2 <=
                            layout.coord_from_index(index)[0] - layout.coord_from_index(index)[1]
                            <= coord_diff]
        
        # Updating layout basis type
        layout.basis = 'dual'
    else:
        raise Exception('Select "primary" or "dual" basis.')

    # Adding links and data qubits to layout object
    layout.observable_links = obs_links
    layout.observable_data_qubits = obs_data_qubits
# End set_diamond_observable
