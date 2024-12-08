# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

class measurement_round():
    '''
    Stores measurement data for a single measurement round including 
    what was measured and when
    '''
    def __init__(self, start_mmt, mmts, label=None):
        self.start_mmt = start_mmt
        self.mmts = mmts
        self.label = label
    # End __init__

    def __str__(self):
        return f'{self.label}: starting at {self.start_mmt} - {self.mmts}'

### End measurement_round class ###

class measurement_tracker():
    '''
    Create an object to store previous measurement data for easy calling
    when defining observables.
    '''
    def __init__(self):
        self.num_mmts = 0
        self.mmt_rounds = 0
        self.measurements = {}
    # End __init__


    def __str__(self): 
        return f'{self.num_mmts} measurements in {self.mmt_rounds} rounds'
    

    def add_measurement(self, measurements:list, label:str=None):
        '''
        Adds measurement round to measurement tracker and updates time values
        '''
        mmt_round = measurement_round(self.num_mmts, measurements)
        self.num_mmts += len(measurements)
        self.measurements[self.mmt_rounds] = mmt_round
        self.mmt_rounds += 1
    # End add_measurement


    def get_mmt_targets(self, indexes:list, rd:int):
        '''
        Takes set of indexes, and a specific round to look at. 
        Returns mmt targets for the given indexes in the previous round specified.
        '''
        assert rd < 0, 'Select a round in the past (requires a negative integer).'
        mmt_round = self.get_previous_round(rd)
        start = mmt_round.start_mmt - self.num_mmts
        targets = [mmt_round.mmts.index(i) + start 
                   for i in indexes
                    if i in mmt_round.mmts]
        return targets
    # End get_mmt_targets

    
    def get_previous_round(self, rd:int) -> measurement_round:
        return self.measurements[self.mmt_rounds + rd]
    # End get_previous_round

### End measurement_tracker class ###



