# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from src.layout_planar_main import Planar_Layout, index_from_coord
from stim import Circuit
import matplotlib.pyplot as plt
import matplotlib as mpl


class Mapping():

    def __init__(self, device_size, delta_x=0, delta_y=0):
        self.device_size = device_size
        self.qubits = []
        self.delta_x, self.delta_y = delta_x, delta_y
    # End __init__

    def add_qubit(self,
                  stim_index:int, 
                  register_index:int, 
                  qiskit_index:int):
        '''
        Takes a set of indexes and adds it to the mapping object.
        '''
        self.qubits.append({'stim_index':stim_index,
                            'register_index':register_index,
                            'qiskit_index':qiskit_index})
    # End add_qubit

    def get_initial_layout(self):
        '''
        Creates list to feed into transpile.
        List index is register index, list value is device location index (qiskit index).
        '''
        return [self.get_qiskit_index(register_index=i) for i in range(len(self.qubits))]
    # End get_initial_layout

    def get_stim_index(self, register_index=None, qiskit_index=None):
        if register_index is not None:
            stim_indexes = [qubit['stim_index'] 
                            for qubit in self.qubits 
                            if qubit['register_index'] == register_index]
            assert len(stim_indexes) == 1, f'There are {len(stim_indexes)} qubits associated with {register_index} register_index.'
            stim_index = stim_indexes[0]
        elif qiskit_index is not None:
            stim_indexes = [qubit['stim_index'] 
                            for qubit in self.qubits 
                            if qubit['qiskit_index'] == qiskit_index]
            assert len(stim_indexes) == 1, f'There are {len(stim_indexes)} qubits associated with {qiskit_index} qiskit_index.'
            stim_index = stim_indexes[0]
        else:
            raise Exception('Please pass a register or qiskit index.')
        return stim_index
    # End get_stim_index

    def get_register_index(self, stim_index=None, qiskit_index=None):
        if stim_index is not None:
            register_indexes = [qubit['register_index'] 
                            for qubit in self.qubits 
                            if qubit['stim_index'] == stim_index]
            assert len(register_indexes) == 1, f'There are {len(register_indexes)} qubits associated with {stim_index} stim_index.'
            register_index = register_indexes[0]
        elif qiskit_index is not None:
            register_indexes = [qubit['register_index'] 
                            for qubit in self.qubits 
                            if qubit['qiskit_index'] == qiskit_index]
            assert len(register_indexes) == 1, f'There are {len(register_indexes)} qubits associated with {qiskit_index} qiskit_index.'
            register_index = register_indexes[0]
        else:
            raise Exception('Please pass a register or qiskit index.')
        return register_index
    # End get_stim_index

    def get_qiskit_index(self, register_index=None, stim_index=None):
        if register_index is not None:
            qiskit_indexes = [qubit['qiskit_index'] 
                            for qubit in self.qubits 
                            if qubit['register_index'] == register_index]
            assert len(qiskit_indexes) == 1, f'There are {len(qiskit_indexes)} qubits associated with {register_index} register_index.'
            qiskit_index = qiskit_indexes[0]
        elif stim_index is not None:
            qiskit_indexes = [qubit['qiskit_index'] 
                            for qubit in self.qubits 
                            if qubit['stim_index'] == stim_index]
            assert len(qiskit_indexes) == 1, f'There are {len(qiskit_indexes)} qubits associated with {stim_index} stim_index.'
            qiskit_index = qiskit_indexes[0]
        else:
            raise Exception('Please pass a register or qiskit index.')
        return qiskit_index
    # End get_stim_index

### End Mapping() class ###


def get_qiskit_mapping(stim_circuit:Circuit, 
                       layout:Planar_Layout,
                       given_stim_index: int, 
                       given_qiskit_index: int, 
                       reflect:bool,
                       device_size=127) -> Mapping:
    '''
    
    '''
    # Defining inputs
    device_coords = get_device_coords(device_size)
    stim_coords = stim_circuit.get_final_qubit_coordinates()

    # Defining coordinate shift
    given_qx, given_qy = device_coords[given_qiskit_index]
    given_sx, given_sy = layout.coord_from_index(given_stim_index)
    if reflect:
        given_sx, given_sy = given_sy, given_sx
    delta_x = given_qx - given_sx
    delta_y = given_qy - given_sy

    # Creating mapping information
    map = Mapping(device_size, delta_x, delta_y)
    register_index = 0
    for stim_index in stim_coords:
        if reflect:
            sy, sx = stim_coords[stim_index]
        else:
            sx, sy = stim_coords[stim_index]
        qx, qy = sx + delta_x, sy + delta_y
        try:
            qiskit_index = device_coords.index([qx, qy])
        except:
            raise ValueError(f'[{qx}, {qy}] is not associated with a qubit on the device.')

        map.add_qubit(stim_index, register_index, qiskit_index)
        register_index += 1
    
    return map
# End get_qiskit_mapping
    

def check_alignment(map:Mapping, 
                    layout:Planar_Layout, 
                    highlights=[], 
                    reflected=True,
                    heatmap={},
                    title='',
                    cmap_key='Reds',
                    noise_coloring='half'):
    '''
    Plots a visual check to ensure that mapping created 
    by get_qiskit_mapping is alligned as intended.
    '''
    # Plotting empty circles for existing qubits on device
    fig, ax = plt.subplots()
    for x,y in get_device_coords(map.device_size):
        ax.plot(y,-x,'o', markersize=8, fillstyle='none', color='tab:blue', zorder=2)

    # Filling circuits for qubits included in mapping
    for qubit in map.qubits:
        sx, sy = layout.coord_from_index(qubit['stim_index'])
        x = sy + map.delta_x if reflected else sx + map.delta_x
        y = sx + map.delta_y if reflected else sy + map.delta_y
        ax.plot(y,-x, 'o', markersize=5, color='tab:red', zorder=3)

    # Heatmap
    cmap = mpl.colormaps[cmap_key] # Blues for m plaqs (orange/purple?)
    for key in heatmap:
        x_str, y_str, anyon = key[1:-1].split(',')
        stim_x, stim_y = int(float(x_str)), int(float(y_str))
        # I have layout, possibly get data_qubits included in 
        x = stim_y + map.delta_x if reflected else stim_x + map.delta_x
        y = stim_x + map.delta_y if reflected else stim_y + map.delta_y
        # Having cmap entry here be weird to get more color distinction near 0
        # Possibly reverse later, just use value instead.
        value = round(heatmap[key], 3)
        # shifted circle eqn for low noise to get better definition
        if noise_coloring == 'circle':
            cmap_value = (1 - (value - 1) ** 2) ** .5 
        elif noise_coloring == 'linear':
            cmap_value = value
        elif noise_coloring == 'half':
            cmap_value = 2 * value if value <= .5 else 1.0
        else:
            raise NameError(f'"{noise_coloring}" is not a valid noise_coloring input.')
        
        text_color = 'black' if cmap_value < .85 else 'white'
        ax.text(y - .6, -x - .2, str(value), color=text_color)
        #ax.add_patch(mpl.patches.Rectangle((y - 2, -x - 1), 4, 2, color=cmap(cmap_value), ))
        ax.add_patch(get_detector_patch(layout, 
                                        stim_x, stim_y, 
                                        map.delta_x, map.delta_y, 
                                        cmap(cmap_value)))

    # Highlights
    for x,y in highlights:
        ax.plot(y,-x,'x', color='tab:green', zorder=1, markersize=10)

    # Final plotting
    plt.xlabel('y')
    plt.ylabel('-x')
    plt.title(title)
    plt.show()
# End check_allignment

def get_detector_patch(layout:Planar_Layout, det_x, det_y, delta_x, delta_y, color):
    '''
    Takes layout and location of a detector, finds all data qubits 
    associated with the detectors, and returns a mpl polygon 
    that represents the detector in space.
    '''
    # Get data_qubits
    det_index = index_from_coord(det_x, det_y, layout.width)
    if det_index in layout.plaq_dict:
        plaq = layout.plaq_dict[det_index]
    elif det_index in layout.truncated_plaquettes:
        plaq = layout.truncated_plaquettes[det_index]
    else:
        raise IndexError(f'({det_x}, {det_y}) is not a valid detector coordinate.')
    
    # Putting together coords in patch
    qubit_coords = [(layout.coord_from_index(index)) 
                    for index in plaq.data_qubits]
    shifted_qubit_coords = [(x + delta_y, -(y + delta_x)) for (x, y) in qubit_coords]
    patch = mpl.patches.Polygon(shifted_qubit_coords, color=color)
    return patch
# End get_detector_patch


def get_device_coords(key):
    qubit_coordinates_map = {}

    qubit_coordinates_map[1] = [[0, 0]]

    qubit_coordinates_map[5] = [[1, 0], [0, 1], [1, 1], [1, 2], [2, 1]]

    qubit_coordinates_map[7] = [[0, 0], [0, 1], [0, 2], [1, 1], [2, 0], [2, 1], [2, 2]]

    qubit_coordinates_map[20] = [
        [0, 0],
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [1, 0],
        [1, 1],
        [1, 2],
        [1, 3],
        [1, 4],
        [2, 0],
        [2, 1],
        [2, 2],
        [2, 3],
        [2, 4],
        [3, 0],
        [3, 1],
        [3, 2],
        [3, 3],
        [3, 4],
    ]

    qubit_coordinates_map[15] = [
        [0, 0],
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [0, 5],
        [0, 6],
        [1, 7],
        [1, 6],
        [1, 5],
        [1, 4],
        [1, 3],
        [1, 2],
        [1, 1],
        [1, 0],
    ]

    qubit_coordinates_map[16] = [
        [1, 0],
        [1, 1],
        [2, 1],
        [3, 1],
        [1, 2],
        [3, 2],
        [0, 3],
        [1, 3],
        [3, 3],
        [4, 3],
        [1, 4],
        [3, 4],
        [1, 5],
        [2, 5],
        [3, 5],
        [1, 6],
    ]

    qubit_coordinates_map[27] = [
        [1, 0],
        [1, 1],
        [2, 1],
        [3, 1],
        [1, 2],
        [3, 2],
        [0, 3],
        [1, 3],
        [3, 3],
        [4, 3],
        [1, 4],
        [3, 4],
        [1, 5],
        [2, 5],
        [3, 5],
        [1, 6],
        [3, 6],
        [0, 7],
        [1, 7],
        [3, 7],
        [4, 7],
        [1, 8],
        [3, 8],
        [1, 9],
        [2, 9],
        [3, 9],
        [3, 10],
    ]

    qubit_coordinates_map[28] = [
        [0, 2],
        [0, 3],
        [0, 4],
        [0, 5],
        [0, 6],
        [1, 2],
        [1, 6],
        [2, 0],
        [2, 1],
        [2, 2],
        [2, 3],
        [2, 4],
        [2, 5],
        [2, 6],
        [2, 7],
        [2, 8],
        [3, 0],
        [3, 4],
        [3, 8],
        [4, 0],
        [4, 1],
        [4, 2],
        [4, 3],
        [4, 4],
        [4, 5],
        [4, 6],
        [4, 7],
        [4, 8],
    ]

    qubit_coordinates_map[53] = [
        [0, 2],
        [0, 3],
        [0, 4],
        [0, 5],
        [0, 6],
        [1, 2],
        [1, 6],
        [2, 0],
        [2, 1],
        [2, 2],
        [2, 3],
        [2, 4],
        [2, 5],
        [2, 6],
        [2, 7],
        [2, 8],
        [3, 0],
        [3, 4],
        [3, 8],
        [4, 0],
        [4, 1],
        [4, 2],
        [4, 3],
        [4, 4],
        [4, 5],
        [4, 6],
        [4, 7],
        [4, 8],
        [5, 2],
        [5, 6],
        [6, 0],
        [6, 1],
        [6, 2],
        [6, 3],
        [6, 4],
        [6, 5],
        [6, 6],
        [6, 7],
        [6, 8],
        [7, 0],
        [7, 4],
        [7, 8],
        [8, 0],
        [8, 1],
        [8, 2],
        [8, 3],
        [8, 4],
        [8, 5],
        [8, 6],
        [8, 7],
        [8, 8],
        [9, 2],
        [9, 6],
    ]

    qubit_coordinates_map[65] = [
        [0, 0],
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [0, 5],
        [0, 6],
        [0, 7],
        [0, 8],
        [0, 9],
        [1, 0],
        [1, 4],
        [1, 8],
        [2, 0],
        [2, 1],
        [2, 2],
        [2, 3],
        [2, 4],
        [2, 5],
        [2, 6],
        [2, 7],
        [2, 8],
        [2, 9],
        [2, 10],
        [3, 2],
        [3, 6],
        [3, 10],
        [4, 0],
        [4, 1],
        [4, 2],
        [4, 3],
        [4, 4],
        [4, 5],
        [4, 6],
        [4, 7],
        [4, 8],
        [4, 9],
        [4, 10],
        [5, 0],
        [5, 4],
        [5, 8],
        [6, 0],
        [6, 1],
        [6, 2],
        [6, 3],
        [6, 4],
        [6, 5],
        [6, 6],
        [6, 7],
        [6, 8],
        [6, 9],
        [6, 10],
        [7, 2],
        [7, 6],
        [7, 10],
        [8, 1],
        [8, 2],
        [8, 3],
        [8, 4],
        [8, 5],
        [8, 6],
        [8, 7],
        [8, 8],
        [8, 9],
        [8, 10],
    ]

    qubit_coordinates_map[127] = [
        [0, 0],
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [0, 5],
        [0, 6],
        [0, 7],
        [0, 8],
        [0, 9],
        [0, 10],
        [0, 11],
        [0, 12],
        [0, 13],
        [1, 0],
        [1, 4],
        [1, 8],
        [1, 12],
        [2, 0],
        [2, 1],
        [2, 2],
        [2, 3],
        [2, 4],
        [2, 5],
        [2, 6],
        [2, 7],
        [2, 8],
        [2, 9],
        [2, 10],
        [2, 11],
        [2, 12],
        [2, 13],
        [2, 14],
        [3, 2],
        [3, 6],
        [3, 10],
        [3, 14],
        [4, 0],
        [4, 1],
        [4, 2],
        [4, 3],
        [4, 4],
        [4, 5],
        [4, 6],
        [4, 7],
        [4, 8],
        [4, 9],
        [4, 10],
        [4, 11],
        [4, 12],
        [4, 13],
        [4, 14],
        [5, 0],
        [5, 4],
        [5, 8],
        [5, 12],
        [6, 0],
        [6, 1],
        [6, 2],
        [6, 3],
        [6, 4],
        [6, 5],
        [6, 6],
        [6, 7],
        [6, 8],
        [6, 9],
        [6, 10],
        [6, 11],
        [6, 12],
        [6, 13],
        [6, 14],
        [7, 2],
        [7, 6],
        [7, 10],
        [7, 14],
        [8, 0],
        [8, 1],
        [8, 2],
        [8, 3],
        [8, 4],
        [8, 5],
        [8, 6],
        [8, 7],
        [8, 8],
        [8, 9],
        [8, 10],
        [8, 11],
        [8, 12],
        [8, 13],
        [8, 14],
        [9, 0],
        [9, 4],
        [9, 8],
        [9, 12],
        [10, 0],
        [10, 1],
        [10, 2],
        [10, 3],
        [10, 4],
        [10, 5],
        [10, 6],
        [10, 7],
        [10, 8],
        [10, 9],
        [10, 10],
        [10, 11],
        [10, 12],
        [10, 13],
        [10, 14],
        [11, 2],
        [11, 6],
        [11, 10],
        [11, 14],
        [12, 1],
        [12, 2],
        [12, 3],
        [12, 4],
        [12, 5],
        [12, 6],
        [12, 7],
        [12, 8],
        [12, 9],
        [12, 10],
        [12, 11],
        [12, 12],
        [12, 13],
        [12, 14],
    ]

    qubit_coordinates_map[133] = [
        [0, 0],
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [0, 5],
        [0, 6],
        [0, 7],
        [0, 8],
        [0, 9],
        [0, 10],
        [0, 11],
        [0, 12],
        [0, 13],
        [0, 14],
        [1, 0],
        [1, 4],
        [1, 8],
        [1, 12],
        [2, 0],
        [2, 1],
        [2, 2],
        [2, 3],
        [2, 4],
        [2, 5],
        [2, 6],
        [2, 7],
        [2, 8],
        [2, 9],
        [2, 10],
        [2, 11],
        [2, 12],
        [2, 13],
        [2, 14],
        [3, 2],
        [3, 6],
        [3, 10],
        [3, 14],
        [4, 0],
        [4, 1],
        [4, 2],
        [4, 3],
        [4, 4],
        [4, 5],
        [4, 6],
        [4, 7],
        [4, 8],
        [4, 9],
        [4, 10],
        [4, 11],
        [4, 12],
        [4, 13],
        [4, 14],
        [5, 0],
        [5, 4],
        [5, 8],
        [5, 12],
        [6, 0],
        [6, 1],
        [6, 2],
        [6, 3],
        [6, 4],
        [6, 5],
        [6, 6],
        [6, 7],
        [6, 8],
        [6, 9],
        [6, 10],
        [6, 11],
        [6, 12],
        [6, 13],
        [6, 14],
        [7, 2],
        [7, 6],
        [7, 10],
        [7, 14],
        [8, 0],
        [8, 1],
        [8, 2],
        [8, 3],
        [8, 4],
        [8, 5],
        [8, 6],
        [8, 7],
        [8, 8],
        [8, 9],
        [8, 10],
        [8, 11],
        [8, 12],
        [8, 13],
        [8, 14],
        [9, 0],
        [9, 4],
        [9, 8],
        [9, 12],
        [10, 0],
        [10, 1],
        [10, 2],
        [10, 3],
        [10, 4],
        [10, 5],
        [10, 6],
        [10, 7],
        [10, 8],
        [10, 9],
        [10, 10],
        [10, 11],
        [10, 12],
        [10, 13],
        [10, 14],
        [11, 2],
        [11, 6],
        [11, 10],
        [11, 14],
        [12, 0],
        [12, 1],
        [12, 2],
        [12, 3],
        [12, 4],
        [12, 5],
        [12, 6],
        [12, 7],
        [12, 8],
        [12, 9],
        [12, 10],
        [12, 11],
        [12, 12],
        [12, 13],
        [12, 14],
        [13, 0],
        [13, 4],
        [13, 8],
        [13, 12]
    ]

    qubit_coordinates_map[433] = [
        [0, 0],
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [0, 5],
        [0, 6],
        [0, 7],
        [0, 8],
        [0, 9],
        [0, 10],
        [0, 11],
        [0, 12],
        [0, 13],
        [0, 14],
        [0, 15],
        [0, 16],
        [0, 17],
        [0, 18],
        [0, 19],
        [0, 20],
        [0, 21],
        [0, 22],
        [0, 23],
        [0, 24],
        [0, 25],
        [1, 0],
        [1, 4],
        [1, 8],
        [1, 12],
        [1, 16],
        [1, 20],
        [1, 24],
        [2, 0],
        [2, 1],
        [2, 2],
        [2, 3],
        [2, 4],
        [2, 5],
        [2, 6],
        [2, 7],
        [2, 8],
        [2, 9],
        [2, 10],
        [2, 11],
        [2, 12],
        [2, 13],
        [2, 14],
        [2, 15],
        [2, 16],
        [2, 17],
        [2, 18],
        [2, 19],
        [2, 20],
        [2, 21],
        [2, 22],
        [2, 23],
        [2, 24],
        [2, 25],
        [2, 26],
        [3, 2],
        [3, 6],
        [3, 10],
        [3, 14],
        [3, 18],
        [3, 22],
        [3, 26],
        [4, 0],
        [4, 1],
        [4, 2],
        [4, 3],
        [4, 4],
        [4, 5],
        [4, 6],
        [4, 7],
        [4, 8],
        [4, 9],
        [4, 10],
        [4, 11],
        [4, 12],
        [4, 13],
        [4, 14],
        [4, 15],
        [4, 16],
        [4, 17],
        [4, 18],
        [4, 19],
        [4, 20],
        [4, 21],
        [4, 22],
        [4, 23],
        [4, 24],
        [4, 25],
        [4, 26],
        [5, 0],
        [5, 4],
        [5, 8],
        [5, 12],
        [5, 16],
        [5, 20],
        [5, 24],
        [6, 0],
        [6, 1],
        [6, 2],
        [6, 3],
        [6, 4],
        [6, 5],
        [6, 6],
        [6, 7],
        [6, 8],
        [6, 9],
        [6, 10],
        [6, 11],
        [6, 12],
        [6, 13],
        [6, 14],
        [6, 15],
        [6, 16],
        [6, 17],
        [6, 18],
        [6, 19],
        [6, 20],
        [6, 21],
        [6, 22],
        [6, 23],
        [6, 24],
        [6, 25],
        [6, 26],
        [7, 2],
        [7, 6],
        [7, 10],
        [7, 14],
        [7, 18],
        [7, 22],
        [7, 26],
        [8, 0],
        [8, 1],
        [8, 2],
        [8, 3],
        [8, 4],
        [8, 5],
        [8, 6],
        [8, 7],
        [8, 8],
        [8, 9],
        [8, 10],
        [8, 11],
        [8, 12],
        [8, 13],
        [8, 14],
        [8, 15],
        [8, 16],
        [8, 17],
        [8, 18],
        [8, 19],
        [8, 20],
        [8, 21],
        [8, 22],
        [8, 23],
        [8, 24],
        [8, 25],
        [8, 26],
        [9, 0],
        [9, 4],
        [9, 8],
        [9, 12],
        [9, 16],
        [9, 20],
        [9, 24],
        [10, 0],
        [10, 1],
        [10, 2],
        [10, 3],
        [10, 4],
        [10, 5],
        [10, 6],
        [10, 7],
        [10, 8],
        [10, 9],
        [10, 10],
        [10, 11],
        [10, 12],
        [10, 13],
        [10, 14],
        [10, 15],
        [10, 16],
        [10, 17],
        [10, 18],
        [10, 19],
        [10, 20],
        [10, 21],
        [10, 22],
        [10, 23],
        [10, 24],
        [10, 25],
        [10, 26],
        [11, 2],
        [11, 6],
        [11, 10],
        [11, 14],
        [11, 18],
        [11, 22],
        [11, 26],
        [12, 0],
        [12, 1],
        [12, 2],
        [12, 3],
        [12, 4],
        [12, 5],
        [12, 6],
        [12, 7],
        [12, 8],
        [12, 9],
        [12, 10],
        [12, 11],
        [12, 12],
        [12, 13],
        [12, 14],
        [12, 15],
        [12, 16],
        [12, 17],
        [12, 18],
        [12, 19],
        [12, 20],
        [12, 21],
        [12, 22],
        [12, 23],
        [12, 24],
        [12, 25],
        [12, 26],
        [13, 0],
        [13, 4],
        [13, 8],
        [13, 12],
        [13, 16],
        [13, 20],
        [13, 24],
        [14, 0],
        [14, 1],
        [14, 2],
        [14, 3],
        [14, 4],
        [14, 5],
        [14, 6],
        [14, 7],
        [14, 8],
        [14, 9],
        [14, 10],
        [14, 11],
        [14, 12],
        [14, 13],
        [14, 14],
        [14, 15],
        [14, 16],
        [14, 17],
        [14, 18],
        [14, 19],
        [14, 20],
        [14, 21],
        [14, 22],
        [14, 23],
        [14, 24],
        [14, 25],
        [14, 26],
        [15, 2],
        [15, 6],
        [15, 10],
        [15, 14],
        [15, 18],
        [15, 22],
        [15, 26],
        [16, 0],
        [16, 1],
        [16, 2],
        [16, 3],
        [16, 4],
        [16, 5],
        [16, 6],
        [16, 7],
        [16, 8],
        [16, 9],
        [16, 10],
        [16, 11],
        [16, 12],
        [16, 13],
        [16, 14],
        [16, 15],
        [16, 16],
        [16, 17],
        [16, 18],
        [16, 19],
        [16, 20],
        [16, 21],
        [16, 22],
        [16, 23],
        [16, 24],
        [16, 25],
        [16, 26],
        [17, 0],
        [17, 4],
        [17, 8],
        [17, 12],
        [17, 16],
        [17, 20],
        [17, 24],
        [18, 0],
        [18, 1],
        [18, 2],
        [18, 3],
        [18, 4],
        [18, 5],
        [18, 6],
        [18, 7],
        [18, 8],
        [18, 9],
        [18, 10],
        [18, 11],
        [18, 12],
        [18, 13],
        [18, 14],
        [18, 15],
        [18, 16],
        [18, 17],
        [18, 18],
        [18, 19],
        [18, 20],
        [18, 21],
        [18, 22],
        [18, 23],
        [18, 24],
        [18, 25],
        [18, 26],
        [19, 2],
        [19, 6],
        [19, 10],
        [19, 14],
        [19, 18],
        [19, 22],
        [19, 26],
        [20, 0],
        [20, 1],
        [20, 2],
        [20, 3],
        [20, 4],
        [20, 5],
        [20, 6],
        [20, 7],
        [20, 8],
        [20, 9],
        [20, 10],
        [20, 11],
        [20, 12],
        [20, 13],
        [20, 14],
        [20, 15],
        [20, 16],
        [20, 17],
        [20, 18],
        [20, 19],
        [20, 20],
        [20, 21],
        [20, 22],
        [20, 23],
        [20, 24],
        [20, 25],
        [20, 26],
        [21, 0],
        [21, 4],
        [21, 8],
        [21, 12],
        [21, 16],
        [21, 20],
        [21, 24],
        [22, 0],
        [22, 1],
        [22, 2],
        [22, 3],
        [22, 4],
        [22, 5],
        [22, 6],
        [22, 7],
        [22, 8],
        [22, 9],
        [22, 10],
        [22, 11],
        [22, 12],
        [22, 13],
        [22, 14],
        [22, 15],
        [22, 16],
        [22, 17],
        [22, 18],
        [22, 19],
        [22, 20],
        [22, 21],
        [22, 22],
        [22, 23],
        [22, 24],
        [22, 25],
        [22, 26],
        [23, 2],
        [23, 6],
        [23, 10],
        [23, 14],
        [23, 18],
        [23, 22],
        [23, 26],
        [24, 1],
        [24, 2],
        [24, 3],
        [24, 4],
        [24, 5],
        [24, 6],
        [24, 7],
        [24, 8],
        [24, 9],
        [24, 10],
        [24, 11],
        [24, 12],
        [24, 13],
        [24, 14],
        [24, 15],
        [24, 16],
        [24, 17],
        [24, 18],
        [24, 19],
        [24, 20],
        [24, 21],
        [24, 22],
        [24, 23],
        [24, 24],
        [24, 25],
        [24, 26],
    ]

    return qubit_coordinates_map[key]
# End device_coords
