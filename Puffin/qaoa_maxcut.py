import numpy as np
import itertools

from qat.lang.AQASM import H
from qat.lang.AQASM import Program
from qat.qpus import PyLinalg as LinAlg
from scipy.optimize import minimize

from qat.lang.AQASM.misc import build_gate
from qat.lang.AQASM import H, RZ, RX, CNOT, QRoutine
from qat.lang.AQASM import Program
from qat.core import Observable, Term

"""Global module parameters used to parametrize experiments"""

logfile = None
method = None
options = None


def set_options(u_logfile, u_method = "Nelder-Mead", u_options={'maxiter':1000, 'disp': False}):
    global logfile
    global method
    global options
    logfile = u_logfile
    method = u_method
    options = u_options
    
"""Data structure for the usecase graph"""    

class SmartChargingGraph:
    
    def __init__(self, priorities, durations):
        
        assert(len(priorities) == len(durations))
        self.V = len(priorities)
        
        self.E = np.minimum(np.outer(priorities, durations), np.outer(durations, priorities))/100
        np.fill_diagonal(self.E, 0)
        
        return
    
    def cut_value(self, cut):
        
        assert(len(cut) == self.V)
        
        cut_mask = np.repeat([cut], len(cut), axis=0)
        return np.sum(self.E[cut_mask != cut_mask.T])/2
    
    @classmethod
    def from_file(cls, filename):
        
        data_file = open(filename, "r")
        jobs = data_file.readlines()[1:]
        durations, priorities = zip(*[tuple(map(int, job.split('\t'))) for job in jobs])
        return cls(priorities, durations)

    
"""Qaoa general routines"""

def mean_op(op, circuit):
    qpu = LinAlg()
    job = circuit.to_job("OBS", observable=op)
    res = qpu.submit(job).value
    return res

def qaoa_circuit(graph, u_B, u_C, l=1):
    
    def param_circ(betas, gammas):

        assert (len(betas) == len(gammas))
        p = len(betas)
        prog = Program()
        qbits = prog.qalloc(graph.V * l)
        
        #Initialization to uniform superposition
        for q in qbits:
            prog.apply(H, q)
                       
        for step in range(p):
            prog.apply(u_C(gammas[step]), qbits)
            prog.apply(u_B(betas[step]), qbits)

        return prog.to_circ()
        
    return param_circ

def qaoa_mean_op(qaoa_p_circ, op):
    
    def obj(angles):
        p = int(len(angles) / 2)
        circ = qaoa_p_circ(angles[:p], angles[p:])
        return mean_op(op, circ)
        
    return obj

"""Qaoa parameter optimization"""

def num_opt_par(objective, init_point):
    opt = minimize(objective, init_point, method=method, options=options)
    p = int(len(init_point) / 2)
    if logfile is not None:
        logfile.write(str(p)+', ' + str(opt.nfev) + '\n')
    
    return opt.x

def interp(objective, angles):

    p = (len(angles) // 2) +1
    coeff = np.arange(0., p)/p

    beta = coeff * np.append([0], angles[:p-1]) + (1. - coeff) * np.append(angles[:p-1], [0])
    gamma = coeff * np.append([0], angles[-p+1:]) + (1. - coeff) * np.append(angles[-p+1:], [0])
    init = np.append(beta, gamma)  
    exp_obj = lambda ang: -objective(ang)
    opt_par = num_opt_par(exp_obj, init)
    
    return opt_par


"""Qaoa for MaxCut"""

#RZZ gate
@build_gate("RZZ", [float], arity=2)
def RZZ(gamma):
    op = QRoutine()
    op.apply(CNOT, [0,1])
    op.apply(RZ(gamma), 1)
    op.apply(CNOT, [0,1])
    
    return op

#Energy operator
def C(graph):
    
    energy = Observable(graph.V)
    for edge in itertools.combinations(range(graph.V), 2):
        energy.add_term(Term(-graph.E[edge]/2, "ZZ", edge))
    energy.constant_coeff += np.sum(graph.E)/4
    
    return energy

#Mixing hamiltonian
def mixing_op(graph):
    
    def u_B(beta):
        op = QRoutine()
        for qbit in range(graph.V):
            op.apply(RX(-beta * 2), qbit)
        return op
            
    return u_B
    
#Phase Hamiltonian
def phase_op(graph):
    
    def u_C(gamma):
        op = QRoutine()
        for qbits in itertools.combinations(range(graph.V), 2):
            op.apply(RZZ(gamma * graph.E[qbits]), qbits)
        return op
    
    return u_C 

#Ideal mean energy function of parameters
def qaoa_mean_energy(graph):
    u_B = mixing_op(graph)
    u_C = phase_op(graph)
    
    def obj(angles):
        if len(angles) == 2:
            return th_energy_mean(graph, angles[0], angles[1])
        else:
            return qaoa_mean_op(qaoa_circuit(graph, u_B, u_C), C(graph))(angles)
        
    return obj

#Analytical expression for mean energy as a function of parameters for p=1 case
def th_edge_mean(graph, edge, beta, gamma):
    u,v = edge
    w = graph.E
    w_u = np.delete(w[u], [u,v])
    w_v = np.delete(w[v], [u,v])
    
    c = np.cos(2*beta)
    s = np.sin(2*beta)
    
    YuZv = np.prod(np.cos(w_u * gamma))
    ZuYv = np.prod(np.cos(w_v * gamma))
    YuYv = np.prod(np.cos((w_u - w_v) * gamma)) - np.prod(np.cos((w_u + w_v) * gamma))
    
    return w[u,v]/2 * (1 + s*c * np.sin(w[u,v]*gamma) * (YuZv + ZuYv) - 1/2 * s**2 * YuYv)

def th_energy_mean(graph, betas, gammas):
    mean = 0
    for edge in itertools.combinations(range(graph.V), 2):
        mean += th_edge_mean(graph, edge, betas, gammas)
    return mean

"""Classical algorithms"""

def wmft_from_cut(instance, cut_val):
    t_list, p_list = instance
    graph = SmartChargingGraph(p_list, t_list)
    return np.sum(np.multiply(p_list, t_list)) + (np.sum(graph.E)/2 - cut_val) * 100

#Brute force algorithm

def brute_force_search(graph, k=2):
    opt_cut_value = 0

    for cut in itertools.product(range(k), repeat=graph.V):
        cut_value = graph.cut_value(cut)
        if cut_value > opt_cut_value:
            opt_cut_value = cut_value
            
    return opt_cut_value

#Randomized algorithm
def rand_expectation(instance):    
    t_list, p_list = instance
    graph = SmartChargingGraph(p_list, t_list)
    return np.sum(np.multiply(p_list, t_list)) + np.sum(graph.E * 100)/4

#Dynamic programm algorithm
def prune(S):
    s = S[0]
    pruned_S = []

    for T, t, b in S[1:]:
        if t == s[1]:
            if T <= s[0]:
                s = (T, t, b)
        else:
            pruned_S.append(s)
            s = (T, t, b)
    pruned_S.append(s)       
    return pruned_S

def dp_exact(instance):
    t_list, p_list = instance
    i_list = np.divide(p_list, t_list)
    n = len(i_list)
    
    ind_sorted = sorted(range(n), key = lambda x: -i_list[x])
    t_sorted = [t_list[i] for i in ind_sorted]
    p_sorted = [p_list[i] for i in ind_sorted]
    
    S = [(p_sorted[0] * t_sorted[0], t_sorted[0], 0)]
    Total_t = t_sorted[0]


    for i in range(1, n):
        A = []
        C = []
        D = []
        for s in S:
            T, t, b = s
            A.append( (T + p_sorted[i]*(t + t_sorted[i]), t + t_sorted[i], 0) )

            if Total_t-t+t_sorted[i] > t:
                C.append( (T + p_sorted[i]*(Total_t - t + t_sorted[i]), Total_t-t+t_sorted[i], 1) )
            else:
                D.append( (T + p_sorted[i]*(Total_t - t + t_sorted[i]), t, 0) )

        Total_t += t_sorted[i]

        C.extend(D)
        A.extend(C)
        S = sorted(A, key = lambda s: s[1] )
        S = prune(S)

    T_opt = min([T for T, _ , _ in S])   

    return T_opt

#QAOA at depth p=1 performance

"""Functions used in experiments"""

#Experimentally observed pattern for initialization of optimal parameter search at depth p=1
def init_pattern(n):
    if n > 19:
        return [0.18, 0.005]
    return [ 0.23, 0.012 - 0.001*(n-6)]
