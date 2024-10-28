# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from qiskit import QuantumCircuit
from qiskit_aer.noise import depolarizing_error, pauli_error
from src.utils import get_stim_circuits
import src.surface_code_decoder as surface_code_decoder
import stim
import pymatching

import numpy as np
import matplotlib.pyplot as plt

class heavyHEX3CX:
    def __init__(self, d:int = 2, T:int = 1, logical_observable:str = 'Z',
                 num_qubits:int = 127, offset:complex = -2+6j, 
                 anc_reset:bool = True, link_reset:bool = False,
                 CXerror:float = 0, Rerror:float = 0, singleQerror:float = 0, idleerror:float = 0,
                 planted_pm_flip:int = -1,
                 skipCX:bool =True):
        
        self.CXerror = CXerror
        self.Rerror = Rerror
        self.singleQerror = singleQerror
        self.num_qubits = num_qubits
        self.d = d
        self.T = T
        self.logical_observable = logical_observable
        self.link_reset = link_reset
        self.skipCX = skipCX

        qubit_coordinates = [offset+k*(1+1j) + i*(1-1j) for k in range(2, 2*d+1, 2) for i in range(2, 2*d+1, 2)]
        unit_cell_coordinates = [offset+k*(1+1j) + i*(1-1j) for k in range(2, 2*d+1, 2) for i in range(2, 2*d+1, 2) if k!=2*d or i !=2]                        
        ancilla_coordinates = [q + 2j for q in unit_cell_coordinates]

        # preparing the dictionary 
        self.q2i = {q: i + d*(i//d) for i, q in enumerate(self.sorted_complex(qubit_coordinates))}
        self.a2i = {a: i - d + d*(i//d) for i, a in enumerate(self.sorted_complex(ancilla_coordinates),start=d)}

        i2q = {v: k for k,v in  self.q2i.items()}
        # i2a = {v: k for k,v in  self.a2i.items()}

        def rot_qpos(qpos,dir=1):
            if dir==1:
                return (qpos-i2q[0])*(1-1j)+2+2j
            elif dir==-1:
                return (qpos-2-2j)*(1+1j)/2+i2q[0]

        q2i_rot = {rot_qpos(k): v for k,v in  self.q2i.items()}
        a2i_rot = {rot_qpos(k): v for k,v in  self.a2i.items()}
        # i2a_rot = {v: k for k,v in  a2i_rot.items()}

        if num_qubits == -1:
            l2i = {(q+a)/2: -1 for q in self.q2i for a in self.a2i if abs(q-a)==2 and (q-a).imag<=0}
            heavyHEX_dict = self.q2i.copy()
            heavyHEX_dict.update(self.a2i)
            if skipCX:
                heavyHEX_dict.update(l2i)
            self.num_qubits = len(heavyHEX_dict)
        else:
            heavyHEX_dict = self.buildHeavyHEX(num_qubits)
            self.num_qubits = num_qubits
        self.heavyHEX_dict = heavyHEX_dict

        qubit_pos = [q for q in q2i_rot.keys()]
        self.qubit_pos = qubit_pos
        qubit_index_list = [heavyHEX_dict[rot_qpos(qpos,-1)] for qpos in qubit_pos]
        self.qubit_index_list = qubit_index_list

        if d%2 == 1:
            anc_posa = [a for a in a2i_rot.keys() if (a.real==4*d and a.imag%8==4) or (a.imag==4*d and a.real%8==0)]
            anc_posb = [a for a in a2i_rot.keys() if (a.real==4*d and a.imag%8==0) or (a.imag==4*d and a.real%8==4)]
        else:
            anc_posb = [a for a in a2i_rot.keys() if (a.real==4*d and a.imag%8==4) or (a.imag==4*d and a.real%8==0)]
            anc_posa = [a for a in a2i_rot.keys() if (a.real==4*d and a.imag%8==0) or (a.imag==4*d and a.real%8==4)]

        ancilla_pos = [a for a in a2i_rot.keys()]
        x_ancilla_pos = [a for a in ancilla_pos if (a.real-2+a.imag-2)%8 == 0]
        z_ancilla_pos = [a for a in ancilla_pos if (a.real-2+a.imag-2)%8 == 4]

        self.measure_x_qubit_pos_cycle0 = [anc_pos for anc_pos in ancilla_pos 
                                    if (anc_pos.real<4*d and anc_pos in z_ancilla_pos)
                                        or anc_pos.imag==4*d]
        measure_x_qubits_cycle0 = [heavyHEX_dict[rot_qpos(anc_pos,-1)] for anc_pos in self.measure_x_qubit_pos_cycle0]
        self.measure_z_qubit_pos_cycle0 = [anc_pos for anc_pos in ancilla_pos 
                                    if (anc_pos.imag<4*d and anc_pos in x_ancilla_pos)
                                        or anc_pos.real==4*d]
        measure_z_qubits_cycle0 = [heavyHEX_dict[rot_qpos(anc_pos,-1)] for anc_pos in self.measure_z_qubit_pos_cycle0]
        self.measure_x_qubit_pos_cycle1 = [anc_pos for anc_pos in ancilla_pos 
                                    if (anc_pos.real<4*d and anc_pos in x_ancilla_pos)
                                        or anc_pos.imag==4*d]
        measure_x_qubits_cycle1 = [heavyHEX_dict[rot_qpos(anc_pos,-1)] for anc_pos in self.measure_x_qubit_pos_cycle1]
        self.measure_z_qubit_pos_cycle1 = [anc_pos for anc_pos in ancilla_pos 
                                    if (anc_pos.imag<4*d and anc_pos in z_ancilla_pos)
                                        or anc_pos.real==4*d]
        measure_z_qubits_cycle1 = [heavyHEX_dict[rot_qpos(anc_pos,-1)] for anc_pos in self.measure_z_qubit_pos_cycle1]


        pair_target_pos_round0 = [[rot_qpos(anc_pos,-1), rot_qpos(anc_pos-2-2j,-1)] for anc_pos in x_ancilla_pos if anc_pos-2-2j in qubit_pos and anc_pos not in anc_posb]
        pair_target_pos_round0.extend([rot_qpos(anc_pos-2-2j,-1), rot_qpos(anc_pos,-1)] for anc_pos in z_ancilla_pos if anc_pos-2-2j in qubit_pos and anc_pos not in anc_posb)

        pair_target_pos_round1 = [[rot_qpos(anc_pos,-1), rot_qpos(anc_pos+2-2j,-1)] for anc_pos in x_ancilla_pos if anc_pos+2-2j in qubit_pos and anc_pos not in anc_posb]
        pair_target_pos_round1.extend([rot_qpos(anc_pos-2+2j,-1), rot_qpos(anc_pos,-1)] for anc_pos in z_ancilla_pos if anc_pos-2+2j in qubit_pos and anc_pos not in anc_posb)

        pair_target_pos_round2 = [[rot_qpos(anc_pos-2+2j,-1), rot_qpos(anc_pos,-1)] for anc_pos in x_ancilla_pos if anc_pos-2+2j in  qubit_pos and anc_pos not in  anc_posa]
        pair_target_pos_round2.extend([rot_qpos(anc_pos,-1), rot_qpos(anc_pos+2-2j,-1)] for anc_pos in z_ancilla_pos if anc_pos+2-2j in  qubit_pos and anc_pos not in  anc_posa)

        pair_target_pos_round3 = [[rot_qpos(anc_pos-2-2j,-1), rot_qpos(anc_pos,-1)] for anc_pos in x_ancilla_pos if anc_pos-2-2j in  qubit_pos and anc_pos not in  anc_posa]
        pair_target_pos_round3.extend([rot_qpos(anc_pos,-1), rot_qpos(anc_pos-2-2j,-1)] for anc_pos in z_ancilla_pos if anc_pos-2-2j in  qubit_pos and anc_pos not in  anc_posa)

        pair_target_pos_rounds = [np.transpose(pair_target_pos_round0),np.transpose(pair_target_pos_round1),np.transpose(pair_target_pos_round2),np.transpose(pair_target_pos_round3)]

        # edge qubits
        edge1_qubit_pos = [k+2j for k in range(2,4*d+2,4)]
        edge2_qubit_pos = [2+k*1j for k in range(2,4*d+2,4)]
        if  logical_observable == 'Z':
            self.edge_qubit_pos = edge1_qubit_pos
        elif  logical_observable == 'X':
            self.edge_qubit_pos = edge2_qubit_pos
        self.edge_qubits =  [heavyHEX_dict[rot_qpos(qpos,-1)] for qpos in self.edge_qubit_pos]


        #build circuit
        qc = QuantumCircuit(self.num_qubits,(d**2-1)*T+d**2)

        #initialize logical
        if Rerror>0: #no reset for backend.run
            self.noisy_reset(qc,qubit_index_list)
        if logical_observable == 'X':
            self.noisy_h(qc,qubit_index_list)
        
        if Rerror>0:
            self.noisy_reset(qc,measure_x_qubits_cycle0)
        self.noisy_h(qc,measure_x_qubits_cycle0)
        if Rerror>0:
            self.noisy_reset(qc,measure_z_qubits_cycle0)
        if not self.link_reset:
            qc.barrier()
            
        measuredict = {} 
        for time in range(T):
            if time%2==1:
                for pair_target_pos in pair_target_pos_rounds:
                    self.skip_CX(qc,pair_target_pos[0],pair_target_pos[1])
                    if not self.link_reset:
                        qc.barrier()
                qc.barrier()

                self.noisy_h(qc,measure_x_qubits_cycle0)

                if planted_pm_flip//(d**2-1)==time:
                    qc.append(pauli_error([("I",0),("X",1)]),[(measure_x_qubits_cycle0+measure_z_qubits_cycle0)[planted_pm_flip%(d**2-1)]])

                self.noisy_measure(qc,measure_x_qubits_cycle0, [time*(d**2-1)+i for i in range(int(np.ceil((d**2-1)/2)))])
                self.noisy_measure(qc,measure_z_qubits_cycle0, [time*(d**2-1)+int(np.ceil((d**2-1)/2))+i for i in range(int((d**2-1)/2))])
                for q in qubit_index_list:
                    qc.append(depolarizing_error(idleerror,1),[q])
                if self.link_reset:
                    qc.barrier()
                if time!= T-1:
                    if anc_reset:
                        self.noisy_reset(qc,measure_x_qubits_cycle0)
                        self.noisy_reset(qc,measure_z_qubits_cycle0)
                    self.noisy_h(qc,measure_x_qubits_cycle0)
                if not self.link_reset and time!=T-1:
                    qc.barrier()
                measuredict.update({
                    (self.measure_x_qubit_pos_cycle0[i],time): time*(d**2-1)+i 
                    for i in range(int(np.ceil((d**2-1)/2)))
                    })
                measuredict.update({
                    (self.measure_z_qubit_pos_cycle0[i],time): time*(d**2-1)+int(np.ceil((d**2-1)/2))+i 
                    for i in range(int((d**2-1)/2))
                    })
            
            elif time%2==0:
                for pair_target_pos in pair_target_pos_rounds[::-1]:
                    self.skip_CX(qc,pair_target_pos[0],pair_target_pos[1])
                    if not self.link_reset:
                        qc.barrier()
                qc.barrier()

                if Rerror>0:
                    for q in measure_x_qubits_cycle1:
                        qc.append(depolarizing_error(Rerror, 1), [q])
                    for q in measure_z_qubits_cycle1:
                        qc.append(depolarizing_error(Rerror, 1), [q])            
                self.noisy_h(qc,measure_x_qubits_cycle1)            

                if planted_pm_flip//(d**2-1)==time:
                    qc.append(pauli_error([("I",0),("X",1)]),[(measure_x_qubits_cycle1+measure_z_qubits_cycle1)[planted_pm_flip%(d**2-1)]])

                self.noisy_measure(qc,measure_x_qubits_cycle1, [time*(d**2-1)+i for i in range(int((d**2-1)/2))])
                self.noisy_measure(qc,measure_z_qubits_cycle1, [time*(d**2-1)+int((d**2-1)/2)+i for i in range(int(np.ceil((d**2-1)/2)))])
                for q in qubit_index_list:
                    qc.append(depolarizing_error(idleerror,1),[q])                
                if self.link_reset:
                    qc.barrier()
                if time!= T-1:
                    if anc_reset:
                        self.noisy_reset(qc,measure_x_qubits_cycle1)
                        self.noisy_reset(qc,measure_z_qubits_cycle1)
                    self.noisy_h(qc,measure_x_qubits_cycle1)
                if not self.link_reset and time!=T-1:
                    qc.barrier()
                measuredict.update({
                    (self.measure_x_qubit_pos_cycle1[i],time): time*(d**2-1)+i 
                    for i in range(int((d**2-1)/2))
                    })
                measuredict.update({
                    (self.measure_z_qubit_pos_cycle1[i],time): time*(d**2-1)+int((d**2-1)/2)+i 
                    for i in range(int(np.ceil((d**2-1)/2)))
                    })

        #final measurements
        if logical_observable == 'X':
            self.noisy_h(qc,qubit_index_list)

        self.noisy_measure(qc,qubit_index_list,[(d**2-1)*T + i for i in range(d**2)])
        measuredict.update({
            (qubit_pos[i],T): (d**2-1)*T+i 
            for i in range(d**2)
            })
        self.circuit = qc
        self.measuredict = measuredict
        self.error_sensitive_events = self.get_error_sensitive_events()

        self.stim_circuit = get_stim_circuits(self.circuit)[0][0]
        for detector in self.error_sensitive_events:
            self.stim_circuit.append("DETECTOR",[stim.target_rec(measind-((d**2-1)*T+d**2)) for measind in detector])
        self.stim_circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(self.measuredict[(pos,T)]-((d**2-1)*T+d**2)) for pos in self.edge_qubit_pos],0)

    def noisy_reset(self,qc,q_list):
        qc.reset(q_list)
        if self.Rerror > 0:
            for q in q_list:
                qc.append(pauli_error([("I",1-self.Rerror),("X",self.Rerror)]),[q])
        pass

    def noisy_measure(self,qc,q_list,c_list):
        if self.Rerror > 0:
            for q in q_list:
                qc.append(pauli_error([("I",1-self.Rerror),("X",self.Rerror)]),[q])
        qc.measure(q_list,c_list)
        pass

    def noisy_h(self,qc,q_list):
        qc.h(q_list)
        if self.singleQerror > 0:
            for q in q_list:
                qc.append(depolarizing_error(self.singleQerror,1),[q])
        pass


    def skip_CX(self, qc,qa_pos_list,qb_pos_list):
        for qa_pos,qb_pos in zip(qa_pos_list,qb_pos_list):
            qa = self.heavyHEX_dict[qa_pos]
            qb = self.heavyHEX_dict[qb_pos]
            if self.skipCX:
                qm = self.heavyHEX_dict[(qa_pos+qb_pos)/2]
                if self.link_reset:
                    self.noisy_reset(qc,qm)
                else:    
                    qc.cx(qm,qb)
                    if self.CXerror>0:
                        qc.append(depolarizing_error(self.CXerror, 2), [qm,qb])
                qc.cx(qa,qm)
                if self.CXerror>0:
                    qc.append(depolarizing_error(self.CXerror, 2), [qa,qm])
                qc.cx(qm,qb)
                if self.CXerror>0:
                    qc.append(depolarizing_error(self.CXerror, 2), [qm,qb])
                qc.cx(qa,qm) #this is only for stim to calm down with the detector error messages...
                if self.CXerror>0:
                    qc.append(depolarizing_error(self.CXerror, 2), [qa,qm])
            else:
                qc.cx(qa,qb)
                if self.CXerror>0:
                    qc.append(depolarizing_error(self.CXerror, 2), [qa,qb])
        pass

    def buildHeavyHEX(self, Nq):
        if Nq == 127:
            rows, cols = 13,15
            HHXlatticepos = [[i,0,i] for i in range(cols-1)]
            i=cols-1

        elif Nq == 133:
            rows, cols = 14,15
            HHXlatticepos = [[i,0,i] for i in range(cols)]
            i=cols

        row = 1
        while row < rows-1:
            col=0
            if row%2==0:
                while col < cols:
                    HHXlatticepos.append([i, row, col])
                    i+=1
                    col+=1
            else:
                while col < cols:
                    if row%4==1 and col%4==0:
                        HHXlatticepos.append([i, row, col])
                        i+=1
                    if row%4==3 and col%4==2:
                        HHXlatticepos.append([i, row, col])
                        i+=1
                    col+=1
            row+=1
        if Nq != 133:    
            HHXlatticepos.extend([[len(HHXlatticepos)+i,rows-1,i+1] for i in range(cols-1)])
        else:
            col=0
            while col < cols:
                if row%4==1 and col%4==0:
                    HHXlatticepos.append([i, row, col])
                    i+=1
                if row%4==3 and col%4==2:
                    HHXlatticepos.append([i, row, col])
                    i+=1
                col+=1
                
        heavyHEX_dict = {qr+qi*1j: i for i,qi,qr in HHXlatticepos}
        return heavyHEX_dict


    # def buildHeavyHEX(self, Nq):
    #     if Nq == 127:
    #         rows, cols = 13,15
    #         # HHXlatticepos = [[i,0,i] for i in range(14)]
    #         # i=14
    #         # row = 1
    #     elif Nq == 433:
    #         rows, cols = 25,27
    #     HHXlatticepos = [[i,0,i] for i in range(cols-1)]
    #     i=cols-1
    #     row = 1
    #     while row < rows-1:
    #         col=0
    #         if row%2==0:
    #             while col < cols:
    #                 HHXlatticepos.append([i, row, col])
    #                 i+=1
    #                 col+=1
    #         else:
    #             while col < cols:
    #                 if row%4==1 and col%4==0:
    #                     HHXlatticepos.append([i, row, col])
    #                     i+=1
    #                 if row%4==3 and col%4==2:
    #                     HHXlatticepos.append([i, row, col])
    #                     i+=1
    #                 col+=1
    #         row+=1

    #     HHXlatticepos.extend([[len(HHXlatticepos)+i,rows-1,i+1] for i in range(cols-1)])

    #     heavyHEX_dict = {qr+qi*1j: i for i,qi,qr in HHXlatticepos}
    #     return heavyHEX_dict

    def sorted_complex(self, xs):
        return sorted(xs, key=lambda v: (v.real+v.imag, v.imag-v.real))
    
    def draw_lattice(self, indices:bool = True):
        qi_list, y_list, x_list = np.transpose([[item[1],item[0].imag,item[0].real] for item in self.heavyHEX_dict.items()])
    
        qubit_index_list, yq_list, xq_list = np.transpose([[item[1],item[0].imag,item[0].real] for item in self.q2i.items()])
        # plt.plot(xq_list,-yq_list,'o')
        # for i,qi in enumerate(qubit_index_list):
        #     plt.text(xq_list[i],-yq_list[i],str(int(qi)))
        ancilla_index_list, ya_list, xa_list = np.transpose([[item[1],item[0].imag,item[0].real] for item in self.a2i.items()])
        # plt.plot(xa_list,-ya_list,'o')
        # for i,ai in enumerate(ancilla_index_list):
        #     plt.text(xa_list[i],-ya_list[i],str(int(ai)))
        
        plt.figure(frameon=False)
        plt.plot(x_list,-y_list,'o',c='gray',alpha = 0.65)
        for i in [int(qi) for qi in qi_list]:
            if indices:
                plt.text(x_list[i],-y_list[i],str(i),fontsize=7)
        plt.plot(xq_list,-yq_list,'o')
        plt.plot(xa_list,-ya_list,'o')
        plt.show()
        pass

    def get_error_sensitive_events(self):
        d = self.d
        T = self.T
        qubit_pos = self.qubit_pos
        blue0_rows = [[re+im*1j for re in range(4,4*d,8)]+[4*d+im*1j] for im in range(8,4*d,8)] #blue if it starts with a complete blue plaquette at meas cycle 0
        red0_rows = [[re+im*1j for re in range(8,4*d,8)]+[4*d+im*1j] for im in range(4,4*d,8)]
        blue0_cols = [[re+im*1j for im in range(8,4*d,8)]+[re+4*d*1j] for re in range(8,4*d,8)]
        red0_cols = [[re+im*1j for im in range(4,4*d,8)]+[re+4*d*1j] for re in range(4,4*d,8)]

        blue1_rows = [[re+im*1j for re in range(4,4*d,8)]+[4*d+im*1j] for im in range(4,4*d,8)] #blue if it starts with a complete blue plaquette at meas cycle 1
        red1_rows = [[re+im*1j for re in range(8,4*d,8)]+[4*d+im*1j] for im in range(8,4*d,8)]
        blue1_cols = [[re+im*1j for im in range(8,4*d,8)]+[re+4*d*1j] for re in range(4,4*d,8)]
        red1_cols = [[re+im*1j for im in range(4,4*d,8)]+[re+4*d*1j] for re in range(8,4*d,8)]

        detectors_coord = []
        if d%2==1:
            for time in range(T):
                if time!=0:
                    if time%2==1:
                        bluet_rows = blue0_rows
                        redt_rows = red0_rows
                        bluet_cols = blue0_cols
                        redt_cols = red0_cols
                    elif time%2==0:
                        bluet_rows = blue1_rows
                        redt_rows = red1_rows
                        bluet_cols = blue1_cols
                        redt_cols = red1_cols

                    for blue_row in bluet_rows:
                        detectors_coord.append([(blue_row[0],time)])
                        for blue_pos in blue_row[1:-1]:
                            detectors_coord.append([(blue_pos-4,time-1),(blue_pos,time)]) 
                        detectors_coord.append([(blue_row[-1]-4,time-1),(blue_row[-1],time-1),(blue_row[-1],time)])
                    for red_row in redt_rows:
                        for blue_pos in red_row[:-1]:
                            detectors_coord.append([(blue_pos-4,time-1),(blue_pos,time)])
                        detectors_coord.append([(red_row[-1],time-1),(red_row[-1],time)])
                    for blue_col in bluet_cols:
                        for red_pos in blue_col[:-1]:
                            detectors_coord.append([(red_pos-4j,time-1),(red_pos,time)])
                        detectors_coord.append([(blue_col[-1],time-1),(blue_col[-1],time)])
                    for red_col in redt_cols:
                        detectors_coord.append([(red_col[0],time)])
                        for red_pos in red_col[1:-1]:
                            detectors_coord.append([(red_pos-4j,time-1),(red_pos,time)])
                        detectors_coord.append([(red_col[-1]-4j,time-1),(red_col[-1],time-1),(red_col[-1],time)])
                
                if time==0 and self.logical_observable=='Z':
                    for blue_row in blue1_rows:
                        detectors_coord.append([(blue_row[0],time)])
                        for blue_pos in blue_row[1:-1]:
                            detectors_coord.append([(blue_pos,time)])
                        detectors_coord.append([(blue_row[-1],time)])
                    for red_row in red1_rows:
                        for blue_pos in red_row[:-1]:
                            detectors_coord.append([(blue_pos,time)])
                        detectors_coord.append([(red_row[-1],time)])
                if time==0 and self.logical_observable=='X':
                    for blue_col in blue1_cols:
                        for red_pos in blue_col[:-1]:
                            detectors_coord.append([(red_pos,time)])
                        detectors_coord.append([(blue_col[-1],time)])
                    for red_col in red1_cols:
                        detectors_coord.append([(red_col[0],time)])
                        for red_pos in red_col[1:-1]:
                            detectors_coord.append([(red_pos,time)])
                        detectors_coord.append([(red_col[-1],time)])
            #final measurement detectors
            if self.logical_observable == 'Z':
                if T%2==0:
                    bluet_rows = blue0_rows
                    redt_rows = red0_rows
                elif T%2==1:
                    bluet_rows = blue1_rows
                    redt_rows = red1_rows
                for blue_row in bluet_rows:
                    for blue_pos in blue_row[:-1]:
                        detector = []
                        detector.append((blue_pos,T-1))
                        for rel_pos in [2+2j,2-2j,-2+2j,-2-2j]:
                            detector.append((blue_pos+rel_pos,T))
                        detectors_coord.append(detector)
                    detectors_coord.append([(blue_row[-1],T-1),(blue_row[-1]-2-2j,T),(blue_row[-1]-2+2j,T)])
                for red_row in redt_rows:
                    detectors_coord.append([(red_row[0]-6-2j,T),(red_row[0]-6+2j,T)])
                    for blue_pos in red_row[:-1]:
                        detector = []
                        detector.append((blue_pos,T-1))
                        if blue_pos==red_row[-2]:
                            detector.append((blue_pos+4,T-1))
                        for rel_pos in [2+2j,2-2j,-2+2j,-2-2j]:
                            detector.append((blue_pos+rel_pos,T))
                        detectors_coord.append(detector)
            elif self.logical_observable == 'X':
                if T%2==0:
                    bluet_cols = blue0_cols
                    redt_cols = red0_cols
                elif T%2==1:
                    bluet_cols = blue1_cols
                    redt_cols = red1_cols
                for blue_col in bluet_cols:
                    detectors_coord.append([(blue_col[0]-2-6j,T),(blue_col[0]+2-6j,T)])
                    for red_pos in blue_col[:-1]:
                        detector = []
                        detector.append((red_pos,T-1))
                        if red_pos == blue_col[-2]:
                            detector.append((red_pos+4j,T-1))
                        for rel_pos in [2+2j,2-2j,-2+2j,-2-2j]:
                            detector.append((red_pos+rel_pos,T))
                        detectors_coord.append(detector)
                for red_col in redt_cols:
                    for red_pos in red_col[:-1]:
                        detector = []
                        detector.append((red_pos,T-1))
                        for rel_pos in [2+2j,2-2j,-2+2j,-2-2j]:
                            detector.append((red_pos+rel_pos,T))
                        detectors_coord.append(detector)
                    detectors_coord.append([(red_col[-1],T-1),(red_col[-1]-2-2j,T),(red_col[-1]+2-2j,T)])

        elif d%2==0:
            for time in range(T):
                if time!=0:
                    if time%2==1:
                        bluet_rows = blue0_rows
                        redt_rows = red0_rows
                        bluet_cols = blue0_cols
                        redt_cols = red0_cols
                    elif time%2==0:
                        bluet_rows = blue1_rows
                        redt_rows = red1_rows
                        bluet_cols = blue1_cols
                        redt_cols = red1_cols
                    for blue_row in bluet_rows:
                        detectors_coord.append([(blue_row[0],time)])
                        for blue_pos in blue_row[1:-1]:
                            detectors_coord.append([(blue_pos-4,time-1),(blue_pos,time)]) 
                        detectors_coord.append([(blue_row[-1],time-1),(blue_row[-1],time)])
                    for red_row in redt_rows:
                        for blue_pos in red_row[:-1]:
                            detectors_coord.append([(blue_pos-4,time-1),(blue_pos,time)])
                        detectors_coord.append([(red_row[-1]-4,time-1),(red_row[-1],time-1),(red_row[-1],time)])
                    for blue_col in bluet_cols:
                        for red_pos in blue_col[:-1]:
                            detectors_coord.append([(red_pos-4j,time-1),(red_pos,time)])
                        detectors_coord.append([(blue_col[-1]-4j,time-1),(blue_col[-1],time-1),(blue_col[-1],time)])
                    for red_col in redt_cols:
                        detectors_coord.append([(red_col[0],time)])
                        for red_pos in red_col[1:-1]:
                            detectors_coord.append([(red_pos-4j,time-1),(red_pos,time)])
                        detectors_coord.append([(red_col[-1],time-1),(red_col[-1],time)])

                if time==0 and self.logical_observable=='Z':
                    for blue_row in blue1_rows:
                        for blue_pos in blue_row:
                            detectors_coord.append([(blue_pos,0)]) 
                    for red_row in red1_rows:
                        for blue_pos in red_row:
                            detectors_coord.append([(blue_pos,0)]) 
                if time==0 and self.logical_observable=='X':
                    for blue_col in blue1_cols:
                        for red_pos in blue_col:
                            detectors_coord.append([(red_pos,0)]) 
                    for red_col in red1_cols:
                        for red_pos in red_col:
                            detectors_coord.append([(red_pos,0)]) 
            #final measurement detectors
            if self.logical_observable == 'Z':
                if T%2==0:
                    bluet_rows = blue0_rows
                    redt_rows = red0_rows
                elif T%2==1:
                    bluet_rows = blue1_rows
                    redt_rows = red1_rows
                for blue_row in bluet_rows:
                    for blue_pos in blue_row[:-1]:
                        detector = []
                        detector.append((blue_pos,T-1))
                        if blue_pos==blue_row[-2]:
                            detector.append((blue_pos+4,T-1))
                        for rel_pos in [2+2j,2-2j,-2+2j,-2-2j]:
                            detector.append((blue_pos+rel_pos,T))
                        detectors_coord.append(detector)
                for red_row in redt_rows:
                    detectors_coord.append([(red_row[0]-6-2j,T),(red_row[0]-6+2j,T)])
                    for blue_pos in red_row[:-1]:
                        detector = []
                        detector.append((blue_pos,T-1))
                        for rel_pos in [2+2j,2-2j,-2+2j,-2-2j]:
                            detector.append((blue_pos+rel_pos,T))
                        detectors_coord.append(detector)
                    detectors_coord.append([(red_row[-1],T-1),(red_row[-1]-2-2j,T),(red_row[-1]-2+2j,T)])
            elif self.logical_observable == 'X':
                if T%2==0:
                    bluet_cols = blue0_cols
                    redt_cols = red0_cols
                elif T%2==1:
                    bluet_cols = blue1_cols
                    redt_cols = red1_cols
                for blue_col in bluet_cols:
                    detectors_coord.append([(blue_col[0]-2-6j,T),(blue_col[0]+2-6j,T)])
                    for red_pos in blue_col[:-1]:
                        detector = []
                        detector.append((red_pos,T-1))
                        for rel_pos in [2+2j,2-2j,-2+2j,-2-2j]:
                            detector.append((red_pos+rel_pos,T))
                        detectors_coord.append(detector)
                    detectors_coord.append([(blue_col[-1],T-1),(blue_col[-1]-2-2j,T),(blue_col[-1]+2-2j,T)])
                for red_col in redt_cols:
                    for red_pos in red_col[:-1]:
                        detector = []
                        detector.append((red_pos,T-1))
                        if red_pos==red_col[-2]:
                            detector.append((red_pos+4j,T-1))
                        for rel_pos in [2+2j,2-2j,-2+2j,-2-2j]:
                            detector.append((red_pos+rel_pos,T))
                        detectors_coord.append(detector)


        detectors = [] #space-time coordinates to measurement indices
        for det_coords in detectors_coord:
            new_detector = []
            for det_coord in det_coords:
                new_detector.append(self.measuredict[det_coord])
            detectors.append(new_detector)
        return detectors
        
    def detector_likelihood(self, detector_samples, freqs = [], temporal_boundaries = False, spatial_boundaries = False, type = None):
        meas_inv = {v:k for k,v ,in self.measuredict.items()}
        bulk_det_inds = []

        if type==None:
            type=self.logical_observable

        relevant_eses = []
        for ese in self.error_sensitive_events:
            measpos_dict = {v:k for k,v in self.measuredict.items()}
            pos,time = measpos_dict[ese[0]] #if the first measurement is in the right category, the rest from the same det should be as well
            if time%2==1:
                if pos in self.measure_z_qubit_pos_cycle0 and type=='Z':
                    relevant_eses.append(ese)
                elif pos in self.measure_x_qubit_pos_cycle0 and type=='X':
                    relevant_eses.append(ese)
                elif time==self.T and type==self.logical_observable:
                    relevant_eses.append(ese)
            else:
                if pos in self.measure_z_qubit_pos_cycle1 and type=='Z':
                    relevant_eses.append(ese)
                elif pos in self.measure_x_qubit_pos_cycle1 and type=='X':
                    relevant_eses.append(ese)
                elif time==self.T and type==self.logical_observable:
                    relevant_eses.append(ese)

        for i, ese in enumerate(relevant_eses):
            ese_time = 0
            is_at_boundary = False
            for meas_ind in ese:
                pos,time = meas_inv[meas_ind]
                if time>ese_time:
                    ese_time = time
                if pos.real in [4,4*self.d] or pos.imag in [4,4*self.d]:
                    is_at_boundary = True
            if temporal_boundaries or (ese_time!=0 and ese_time!=self.T):
                if spatial_boundaries or not is_at_boundary:
                   bulk_det_inds.append(i)
        det_probs = []
        if freqs == []:
            freqs=[1]*len(detector_samples)
        for sample,freq in zip(detector_samples,freqs):
            if len(bulk_det_inds)!=0:
                det_prob = sum(np.array(sample)[bulk_det_inds])/len(bulk_det_inds)
            else:
                det_prob = 0
            det_probs.append(freq*det_prob)
        return sum(det_probs)/sum(freqs)

    def detector_likelihood_from_DEM(self, temporal_boundaries = False, spatial_boundaries = False, type = None):
        stim_DEM = self.stim_circuit.detector_error_model(decompose_errors=True,approximate_disjoint_errors=True)

        if type==None:
            type=self.logical_observable

        relevant_eses = []
        # relevant_eses = self.error_sensitive_events
        for ese in self.error_sensitive_events:
            measpos_dict = {v:k for k,v in self.measuredict.items()}
            pos,time = measpos_dict[ese[0]] #if the first measurement is in the right category, the rest from the same det should be as well
            if time%2==1:
                if pos in self.measure_z_qubit_pos_cycle0 and type=='Z':
                    relevant_eses.append(ese)
                elif pos in self.measure_x_qubit_pos_cycle0 and type=='X':
                    relevant_eses.append(ese)
                elif time==self.T and type==self.logical_observable:
                    relevant_eses.append(ese)
            else:
                if pos in self.measure_z_qubit_pos_cycle1 and type=='Z':
                    relevant_eses.append(ese)
                elif pos in self.measure_x_qubit_pos_cycle1 and type=='X':
                    relevant_eses.append(ese)
                elif time==self.T and type==self.logical_observable:
                    relevant_eses.append(ese)


        meas_inv = {v:k for k,v ,in self.measuredict.items()}
        bulk_det_inds = []
        for i, ese in enumerate(relevant_eses):
            ese_time = 0
            is_at_boundary = False
            for meas_ind in ese:
                pos,time = meas_inv[meas_ind]
                if time>ese_time:
                    ese_time = time
                if pos.real in [4,4*self.d] or pos.imag in [4,4*self.d]:
                    is_at_boundary = True
            if temporal_boundaries or (ese_time!=0 and ese_time!=self.T):
                if spatial_boundaries or not is_at_boundary:
                    bulk_det_inds.append(i)

        det_probs={}
        for instruction in stim_DEM:
            if isinstance(instruction, stim.DemInstruction):
                if instruction.type == "error":
                    p = instruction.args_copy()[0]
                    for t in instruction.targets_copy():
                        if t.is_relative_detector_id():
                            t = t.val
                            if t in bulk_det_inds:
                                if t in det_probs:
                                    old_p = det_probs[t]
                                    new_p = old_p*(1-p)+p*(1-old_p)
                                    det_probs[t] = new_p
                                else:
                                    det_probs[t] = p
        if len(det_probs)==0:
            det_likelihood = 0
        else:
            det_likelihood = sum(det_probs.values())/len(det_probs)
        return det_likelihood


    def matching(self):
        return surface_code_decoder.detector_error_model_to_pymatching_graph(self.stim_circuit.detector_error_model(decompose_errors=True,approximate_disjoint_errors=True))
    
    def PredictedObservableOutcome(self, sample, m: pymatching.Matching):
        return m.decode(sample)[0]