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
from termcolor import cprint


class Planar_Layout():
    '''
    Creating layout for a planar code. 
    '''
    def __init__(self, 
                 plaq_indexes, 
                 width, 
                 noise=1e-3*np.ones(5), 
                 color_equals_pauli=True,
                 pauli_cycle=0,
                 noise_dict=None):
        '''
        Initializing Layout object given input parameters. 
        '''
        # Defining basics
        self.width = width
        self.color_equals_pauli = color_equals_pauli
        self.border = None
        self.template = None
        (self.init_noise, 
        self.idle_noise, 
        self.readout_noise, 
        self.single_gate_noise,
        self.two_gate_noise) = noise


        # Creating bulk plaquettes and links.
        (self.link_dict,
         self.plaq_dict, 
         self.data_qubits) = create_layout(self, plaq_indexes, width, pauli_cycle)
        
        # Creating noise dict, either from given dictionary or from list
        if noise_dict is None:
            self.noise_dict = noise_dict_from_list(self, noise)
            # TODO: Setting noise values to average of dictionary entries.
        else:
            self.noise_dict = noise_dict

        # Initialize truncated features
        self.truncated_links = {}
        self.truncated_plaquettes = {}

        # Define observable location, 
        # these get set in the template methods for specific layouts
        self.observable_links = [] 
        self.observable_data_qubits = []
        self.basis = 'None'
        self.border = 'None'
    # End __init__

    def coord_from_index(self, index):
        '''
        Returns x and y coord for an index in the layout.
        '''
        width = self.width
        x = index % width
        y = index // width
        return x, y
    # End coord_from_index

    def index_from_coord(self, x, y):
        '''
        Returns x and y coord for an index in the layout.
        '''
        return y * self.width + x
    # End coord_from_index

    def print(self, observable=False, highlights=[], borders=False):
        '''
        Prints a diagram of the layout (x,y reflected)
        First collect feature elements with 
        key = index : value = (text, color, highlight)
        '''
        # Collecting features
        colors = ('red', 'blue', 'green')
        on_colors = ('on_red', 'on_blue', 'on_green')
        features = {}
        #border_plaqs = planar_codes.layout_borders.get_border_plaquettes()
        
        # Links
        for l in self.link_dict:
            link = self.link_dict[l]
            features[link.index] = (link.pauli_type, colors[link.color_type], None)

        # Plaquettes
        for p in self.plaq_dict:
            plaq = self.plaq_dict[p]
            text = ' '#B' if plaq.index in border_plaqs else ' '
            features[plaq.index] = (text, 'black', on_colors[plaq.color_type])

        # Truncated links
        for tl in self.truncated_links:
            link = self.truncated_links[tl]
            features[link.index] = ('t', colors[link.color_type], None)

        # Truncated Plaquettes
        for tp in self.truncated_plaquettes:
            plaq = self.truncated_plaquettes[tp]
            if not borders:
                features[plaq.index] = ('T', 'white', on_colors[plaq.color_type])
            else: 
                if plaq.anyon_type == 'm':
                    features[plaq.index] = ('m', 'white', on_colors[plaq.color_type])
                elif plaq.anyon_type == 'e':
                    features[plaq.index] = ('e', 'white', on_colors[plaq.color_type])
                elif plaq.anyon_type == 'b':
                    features[plaq.index] = ('b', 'white', on_colors[plaq.color_type])

        # Data Qubits
        for d in self.data_qubits:
            # Defining color
            if observable and d in self.observable_data_qubits:
                color = 'yellow'
            elif d in self.truncated_links:
                color = colors[self.truncated_links[d].color_type]
            else:
                color = None
            # Adding to features
            features[d] = ('+', color, None)

        # Highlights
        for index in highlights:
            if index in features:
                features[index] = (features[index][0], features[index][1], 'on_yellow')
            else:
                features[index] = (' ', None, 'on_yellow')

        # Printing 
        max_point = max(self.data_qubits) + 2 * self.width
        for i in range(max_point):
            if i == 0:
                print('y\\x', end='')
            elif i < self.width:
                print((i+1) % 10, end='')
            elif i % self.width == 0 and i > 0:
                print()
                print((i // self.width) % 10, end='')
            elif i in features:
                cprint(text=features[i][0], 
                       color=features[i][1], 
                       on_color=features[i][2],
                       end='')
            else:
                print(' ', end='')
            print(' ', end='')
        print()
    # End print

### End Planar_Layout() class ###



class Link():
    '''
    
    '''
    def __init__(self, index, width, color_equals_pauli, pauli_cycle, color_type=None, truncated=False):
        '''
        
        '''
        self.index = index
        self.x, self.y = coord_from_index(index, width)
        self.adj0, self.adj1 = self.get_adj_qubits(width)
        self.truncated = truncated

        if color_type is None:
            self.color_type = (get_location_type(self.x, self.y) + pauli_cycle) % 3
        else:
            self.color_type = color_type

        if color_equals_pauli:
            self.pauli_type = ['Z', 'X', 'Y'][self.color_type]
        else:
            if truncated:
                raise Exception("Sorry, Truncated links do not work with orientation style paulis.")
            else:
                self.pauli_type = get_link_orientation_type(self, width)
    # End __init__


    def get_adj_qubits(self, width):
        '''
        Returns data qubits that a link is connected to.
        The layout is designed as a vertially aligned brick layout, so every link
        is either vertically or horizontally oriented depending on which column it is in.
        '''
        # Horizontal link
        if self.y % 2 == 1:
            adj0 = index_from_coord(self.x, self.y - 1, width)
            adj1 = index_from_coord(self.x, self.y + 1, width)
        # Vertial link
        else:
            adj0 = index_from_coord(self.x - 1, self.y, width)
            adj1 = index_from_coord(self.x + 1, self.y, width)

        return adj0, adj1
    # End get_adj_qubits

### End Link() class ###


class Plaquette():
    '''
    Defining a plquette object to be part of a Planar_Layout
    '''
    def __init__(self, index, width, links, truncated=False):
        '''
        Using links around the plaquette to define plaquette.
        Links to be passed here determined in create_layout()
        '''
        self.index = index
        self.x, self.y = coord_from_index(index, width)
        (self.color_type, 
         self.all_links,
         self.data_qubits) = self.sort_links(links)
        self.truncated = truncated
        self.anyon_type = 'b'
    # End __init__


    def sort_links(self, links):
        '''
        Looks at the links creating the plaquette, 
        sorts them and determines the color of the plaquette.
        '''
        # Sorting links
        sorted_links = [0,0,0]
        all_links = []
        data_qubits = []
        for link in links:
            sorted_links[link.color_type] += 1
            all_links.append(link)
            if link.adj0 not in data_qubits and link.adj0 is not None:
                data_qubits.append(link.adj0)
            if link.adj1 not in data_qubits and link.adj1 is not None:
                data_qubits.append(link.adj1)

        # Checking which colors are not present around the plaquette.
        plaq_color = -1
        empty_colors = 0
        for c in range(3):
            if sorted_links[c] == 0:
                plaq_color = c
                empty_colors += 1

        assert_msg = f'When sorting the links for plaquette {self.index}, '
        assert_msg += f'there were {empty_colors} link color types not around the plaquette.'
        assert empty_colors == 1, assert_msg

        #return plaq_color, sorted_links, all_links, data_qubits
        return plaq_color, all_links, data_qubits
    # End sort_links

### End Plaquette() class ###



def create_layout(layout:Planar_Layout, plaq_indexes, width, pauli_cycle):
    '''
    Defines plaquettes and links for a planar layout.
    Also enforces where the plaquettes live in 2D space. 
    This works by looking at every plaquette index, 
    first adding any nonexisting links associated with the plaquette to links, 
    then adding the plaquette to the plaquette various. Then does some sorting.
    '''
    # Create plaquettes and links: index as key, object as value.
    plaquettes = {}
    links = {}
    for index in plaq_indexes:
        # Input assertions
        x, y = coord_from_index(index, width)
        message = f'{index} is not a valid plaquette location for a width {width} layout. '
        assert y % 2 == 1, message + 'Should be centered on an odd y.'
        assert ((y - 1) + x) % 4 == 2, message + 'Not aligned in the x direction.'
        assert x != 0 and x < width - 2, message + 'Too close to boundary.'

        # Defining associated links
        link_indexes = get_adjacent_link_indexes(index, width)
        for link_index in link_indexes:
            # Link does not exist, need to create it.
            if link_index not in links:
                link_obj = Link(link_index, width, layout.color_equals_pauli, pauli_cycle)
                links[link_index] = link_obj
            # Link already exists.
            else:
                link_obj = links[link_index]

        # Now define the plaquette
        adj_links = [links[link_index] for link_index in link_indexes]
        plaquettes[index] = Plaquette(index, width, adj_links)

    # Find data qubits
    data_qubits = []
    for link in links:
        adj0, adj1 = links[link].adj0, links[link].adj1
        if adj0 not in data_qubits:
            data_qubits.append(adj0)
        if adj1 not in data_qubits:
            data_qubits.append(adj1)

    sorted_data_qubits = sorted(data_qubits)

    return links, plaquettes, sorted_data_qubits 
# End create_layout


### HELPER METHODS ###

def get_location_type(x:int, y:int) -> int:
    '''
    Takes index and layout width and returns the link color type:
    0: red, 1: blue, 2: green, -1: data qubit, -2: plaquette index, -3: empty
    This is currently specific to H3 layout
    '''
    location_type = None

    # Vertical Links
    if y % 2 == 0:
        #if x % 2 == 1:
            #link_type = (x + 1) // 2
        if x % 6 == 1:
            location_type = 1
        elif x % 6 == 3:
            location_type = 2
        elif x % 6 == 5:
            location_type = 0
        else: 
            location_type = -1
             
    # Horizontal links
    else:
        # Odd column of plaquettes in coord system from y=0 ->
        if y % 4 == 1:
            #if x % 4 == 0:
                #link_type = 2 - (x // 4) % 3
            if x % 12 == 0:
                location_type = 2
            elif x % 12 == 4:
                location_type = 1
            elif x % 12 == 8:
                location_type = 0
            elif x % 2 == 1:
                location_type = -3
            else: 
                location_type = -2
        # Even column of plaquettes in coord system from y=0 ->
        else: # y % 4 == 3
            #if x % 4 == 2:
                #link_type = ((x - 2) // 4)
            if x % 12 == 2:
                location_type = 0
            elif x % 12 == 6:
                location_type = 2
            elif x % 12 == 10:
                location_type = 1
            elif x % 2 == 1:
                location_type = -3
            else: 
                location_type = -2

    if location_type is not None:
        return location_type
    else:
        raise Exception("This shouldn't happen...")
# End get_aux_type


def is_plaq_index(index, width):
    '''
    Simple call to get_location_type for readability
    '''
    x, y = coord_from_index(index, width)
    return get_location_type(x, y) == -2
# End is_plaq_index


def is_link_index(index, width):
    '''
    Simple call to get_location_type for readability
    '''
    x, y = coord_from_index(index, width)
    return get_location_type(x, y) in (0, 1, 2)
# End is_link_index


def is_data_qubit_index(index, width):
    '''
    Simple call to get_location_type for readability
    '''
    x, y = coord_from_index(index, width)
    return get_location_type(x, y) == -1
# End is_data_qubit_index


def get_link_orientation_type(x:int, y:int):
    '''
    Takes index, norms it to a unit cell, 
    then return pauli type based on edge orientation.
    "-" = Z, "/" = X, "\" = Y
    '''
    assert get_location_type(x, y) in (0,1,2), f'({x},{y}) is not a link location.'
    pauli_type = None
    
    # Plaquette column
    if y % 2 == 1:
        if (x + (y - 1)) % 4 == 0:
            pauli_type = 'Z'

    # Qubit column
    else:
        if ((x - 1) + y) % 4 == 0:
            pauli_type = 'X'
        elif ((x - 1) + y) % 4 == 2:
            pauli_type = 'Y'

    # Check a pauli type was assigned.
    if pauli_type is None:
        msg = f'({x},{y}) was not associated with a pauli_type.'
        msg += '(likely mistake in get_location_type)'
        raise IndexError(msg)
    
    return pauli_type
# End get_pauli_type


def get_adjacent_plaquettes(layout:Planar_Layout, 
                            plaq_index):
    '''
    Takes a plaquette index as an argument and returns all adjacent plaquettes.
    (If they exist.)
    Edge Case: Careful of wrap arounds like in the (2,1) plaquette. 
    '''
    adjacent_plaquettes = []

    plaq = layout.plaq_dict[plaq_index]
    x, y = plaq.x, plaq.y
    w = layout.width
    # Checking which surrounding plaquettes exist
    if index_from_coord(x + 4, y, w) in layout.plaq_dict:         # .
        adjacent_plaquettes.append(index_from_coord(x + 4, y, w))
    if index_from_coord(x - 4, y, w) in layout.plaq_dict:         # ^
        adjacent_plaquettes.append(index_from_coord(x - 4, y, w))
    if index_from_coord(x + 2, y + 2, w) in layout.plaq_dict:     # .>
        adjacent_plaquettes.append(index_from_coord(x + 2, y + 2, w))
    if index_from_coord(x + 2, y - 2, w) in layout.plaq_dict:     # <.
        adjacent_plaquettes.append(index_from_coord(x + 2, y - 2, w))
    if index_from_coord(x - 2, y + 2, w) in layout.plaq_dict:     # ^>
        adjacent_plaquettes.append(index_from_coord(x - 2, y + 2, w))
    if index_from_coord(x - 2, y - 2, w) in layout.plaq_dict:     # <^
        adjacent_plaquettes.append(index_from_coord(x - 2, y - 2, w))
    return adjacent_plaquettes
# End get_adjacent_plaquettes


def get_adjacent_plaquette_locations(layout:Planar_Layout, 
                                     plaq_index):
    '''
    Takes a plaquette index as an argument and returns all adjacent plaquettes.
    (If they exist.)
    Edge Case: Careful of wrap arounds like in the (2,1) plaquette. 
    '''
    adjacent_plaquettes = []

    plaq = layout.plaq_dict[plaq_index]
    x, y = plaq.x, plaq.y
    w = layout.width
    # Adding all adjacent plaquette locations
    adjacent_plaquettes.append(index_from_coord(x + 4, y, w))
    adjacent_plaquettes.append(index_from_coord(x - 4, y, w))
    adjacent_plaquettes.append(index_from_coord(x + 2, y + 2, w))
    adjacent_plaquettes.append(index_from_coord(x + 2, y - 2, w))
    adjacent_plaquettes.append(index_from_coord(x - 2, y + 2, w))
    adjacent_plaquettes.append(index_from_coord(x - 2, y - 2, w))
    return adjacent_plaquettes
# End get_adjacent_plaquettes


def get_adjacent_link_indexes(index, width):
    '''
    Takes plaquette index and returns the indexes
    of all links associated with the plaquette.
    '''
    x, y = coord_from_index(index, width)
    assert get_location_type(x, y) == -2, 'Not a valid plaquette location.'
    link_indexes = [index_from_coord(x + 2, y, width),     # ^
                    index_from_coord(x + 1, y + 1, width), # ^>
                    index_from_coord(x - 1, y + 1, width), # .>
                    index_from_coord(x - 2, y, width),     # .
                    index_from_coord(x - 1, y - 1, width), # <.
                    index_from_coord(x + 1, y - 1, width)] # <^
    
    return link_indexes
# End get_adjacent_link_indexes


def get_adjacent_data_qubit_indexes(index, width):
    '''
    Takes plaquette index and returns the indexes
    of all links associated with the plaquette.
    '''
    x, y = coord_from_index(index, width)
    assert get_location_type(x, y) == -2, 'Not a valid plaquette location.'
    link_indexes = [index_from_coord(x, y + 1, width),     # >
                    index_from_coord(x + 2, y + 1, width), # ^>
                    index_from_coord(x - 2, y + 1, width), # .>
                    index_from_coord(x, y - 1, width),     # <
                    index_from_coord(x - 2, y - 1, width), # <.
                    index_from_coord(x + 2, y - 1, width)] # <^
    
    return link_indexes
# End get_adjacent_link_indexes


def coord_from_index(index, width):
    '''
    Returns x and y coord for an index in the layout.
    NOTE: Possibly set a default width in the future, maybe tied to eagle or falcon.
    '''
    x = index % width
    y = index // width
    return x, y
# End coord_from_index


def index_from_coord(x, y, width):
    '''
    Returns the index for a given coordinate.
    '''
    return x + y * width
# End index_from_coord


def sort_objects(objects):
    '''
    Takes link/plaquette dictionary created in create_layout of all associated objects
    and sorts them into an array with 3 subarrays, once for each color type.
    '''
    sorted_objs = [[],[],[]]
    sorted_keys = sorted(objects)
    for key in sorted_keys:
        color = objects[key].color_type
        sorted_objs[color].append(objects[key])
    return sorted_objs
# End sort_objects


def noise_dict_from_list(layout:Planar_Layout, noise_list):
    '''
    Takes old style length 5 list and converts it to newer dictionary
    input for noise model.
    '''
    all_qubits = layout.data_qubits + [int(qubit) for qubit in layout.link_dict]

    noise_dict = {}
    for qubit in all_qubits:
        noise_dict[qubit] = {'init': noise_list[0],
                             'idle': noise_list[1],
                             'RO'  : noise_list[2],
                             'gate': noise_list[3],
                             '2-gate': {'default': noise_list[4]}}
    return noise_dict
# End noise_dict_from_list
