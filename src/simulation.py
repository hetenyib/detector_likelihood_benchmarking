# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from src.CSS_direct import Hex_Code
from qiskit import transpile
from qiskit_aer import AerSimulator


def error_rate_sim(code:Hex_Code, shots):
    '''
    Runs an error rate simulation for a given Hex_Code object.
    '''
    #time_0 = perf_counter()
    syns, obs = code.sample(shots)
    #time_1 = perf_counter()
    errors = 0
    for s in range(shots):
        errors += code.match.decode(syns[s]) != obs[s]
    #time_2 = perf_counter()
    #print('Sampling time:', time_1 - time_0, '; Decoding time:', time_2 - time_1, 'fraction d/s-', (time_2 - time_1)/(time_1 - time_0))
    return errors[0] / shots
# End error_rate_sim


def run_aer_sim(qiskit_circuit, shots=1024):
    backend = AerSimulator(method='extended_stabilizer')
    qc_compiled = transpile(qiskit_circuit, 
                            backend)

    # Run qiskit sim
    job_sim = backend.run(qc_compiled, shots=shots)
    result_sim = job_sim.result()
    counts = result_sim.get_counts(qc_compiled)

    return counts
# End run_aer_sim
