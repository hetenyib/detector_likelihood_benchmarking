# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

def get_logical_op(sub_round, code_type='H3', basis='primary'):

    if code_type == 'H3':
        logical_op = ['Y_YY_Y',
                  'X_XX_X',
                  '_XX_XX',
                  '_ZZ_ZZ',
                  'ZZ_ZZ_',
                  'YY_YY_'][sub_round % 6]
        
    elif code_type == 'CSS':
        if basis == 'primary':
            logical_op = ['Z_ZZ_Z',
                        'Z_ZZ_Z',
                        '_ZZ_ZZ',
                        '_ZZ_ZZ',
                        'ZZ_ZZ_',
                        'ZZ_ZZ_'][sub_round % 6]
            
        elif basis == 'dual':
            logical_op = [
                        'XX_XX_',
                        '_XX_XX',
                        '_XX_XX',
                        'X_XX_X',
                        'X_XX_X',
                        'XX_XX_'][sub_round % 6]
        
    else:
        raise Exception(f'{code_type} is not included in logical_op()')
        
    return logical_op
# End get_logical_op



def get_logical_frame(sub_round, code_type='H3'):

    if code_type == 'H3':
        logical_frame = ['Y', 'X', 'X', 'Z', 'Z', 'Y'][sub_round % 6]

    else:
        raise Exception(f'{code_type} is not included in logical_op()')
        
    return logical_frame
# End get_logical_frame
