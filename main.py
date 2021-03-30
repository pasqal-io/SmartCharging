import quantum_routines as qr
import parametres

import numpy as np
import json
import scipy
import pathlib

from mpi4py import MPI
import qutip


def main(x):
    parametres.init(x, layers=6, n=False)
    if parametres.problem == 'MIS':
        main_MIS()
    elif parametres.problem == 'MKC':
        main_MKC()


def main_MKC():
    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    p = comm.Get_size()
    indices = splitting(p)
    global no_fev
    no_fev = 0
    main_MKC_2(indices[my_rank])


def main_MIS():
    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    p = comm.Get_size()
    indices = splitting(p)
    global no_fev
    no_fev = 0
    main_connex_MIS_2(indices[my_rank])


def main_MIS_2(index_array):
    for i in index_array:
        Input_string = parametres.data_string + ''. join(str(i)) + ".csv"
        pos = qr.sol_to_pos(Input_string)
        resultz = []
        g = qr.generate_graph(pos)
        (H, indices) = qr.generate_Hilbert_space(g)
        Hams = qr.generate_Hamiltonians(H, indices)
        N = len(pos)
        H_0 = qutip.Qobj(Hams[0])
        H_c = - qutip.Qobj(scipy.sparse.diags(Hams[1], 0))
        H_c_plus = -H_c
        if H_0.dims[0][0] < 3500:
            if parametres.noise:
                Nqubit = N
                c_operator = c_operators(N, Nqubit, H)
            x = [[0] for i in range(H_0.dims[0][0])]
            x[0] = [1]
            psi_0 = qutip.Qobj(x)
            print("Working on", Input_string)
            results = qr.param_evolution_with_layers(qr.func,
                                                     parametres.maxfev,
                                                     parametres.l,
                                                     psi_0,
                                                     H_0,
                                                     H_c,
                                                     H_c_plus)
            resultz.append(results)
            my_data = json.dumps(resultz)
            String = parametres.result_string + ''. join(str(i))
            f = open(String + '.txt', "w+")
            f.close()
            f = open(String + '.txt', "a+")
            f.write(my_data + "\n")
            f.close()


def main_connex_MIS_2(index_array):
    for i in index_array:
        mean_cost = []
        Input_string = parametres.data_string + ''. join(str(i)) + ".csv"
        pos = qr.sol_to_pos(Input_string)
        resultz = []
        g = qr.generate_graph(pos)
        print("Working on", Input_string, ", number of components =",
              len(g.components()))
        for j in g.components():
            jj = g.subgraph(j)
            (H, indices) = qr.generate_Hilbert_space(jj)
            Hams = qr.generate_Hamiltonians(H, indices)
            N = len(pos)
            H_0 = qutip.Qobj(Hams[0])
            H_c = - qutip.Qobj(scipy.sparse.diags(Hams[1], 0))
            H_c_plus = -H_c
            x = [[0] for i in range(H_0.dims[0][0])]
            x[0] = [1]
            psi_0 = qutip.Qobj(x)
            if parametres.noise:
                Nqubit = N
                c_operator = c_operators(N, Nqubit, H)
            paras, sub_cost = qr.param_evolution_with_layers(qr.func,
                                                             parametres.maxfev,
                                                             parametres.l,
                                                             psi_0,
                                                             H_0,
                                                             H_c,
                                                             H_c_plus)
            mean_cost.append(np.array(sub_cost))
        results = []
        cost = np.stack(mean_cost)
        for ii in range(parametres.l):
            results.append(np.mean(cost[:, ii]))
        resultz.append(results)
        my_data = json.dumps(resultz)
        String = parametres.result_string + ''. join(str(i))
        f = open(String + '.txt', "w+")
        f.close()
        f = open(String + '.txt', "a+")
        f.write(my_data + "\n")
        f.close()


def main_MKC_2(index_array):
    for i in index_array:
        Input_string = parametres.data_string + ''. join(str(i)) + ".csv"
        results = []
        A = qr.data_to_adj_matrix(Input_string)
        N = len(A)
        Nqubit = N*2
        results.append(A.tolist())
        H_c = qutip.Qobj(scipy.sparse.diags(
            qr.cost_hamiltonian(A)/np.max(A), 0))
        print(f'H_c is {H_c}')
        H_c_plus = - H_c
        H_0 = qutip.Qobj(qr.H_ref(A))
        psi_0 = qutip.Qobj(
            [[1/(np.sqrt(2**Nqubit))] for i in range(2**Nqubit)])
        if parametres.noise:
            c_operator = c_operators(N, Nqubit, H_0)
        print("Working on", Input_string)
        evolution_w_layers = qr.param_evolution_with_layers(qr.func,
                                                            parametres.maxfev,
                                                            parametres.l,
                                                            psi_0,
                                                            H_0,
                                                            H_c,
                                                            H_c_plus)
        results.append(evolution_w_layers)
        my_data = json.dumps(results)
        String = parametres.result_string + ''. join(str(i))
        f = open(String + '.txt', "w+")
        f.close()
        f = open(String + '.txt', "a+")
        f.write(my_data + "\n")
        f.close()


def splitting(Ncores):
    indexes = []
    for i in range(1, 200):
        Input_string = parametres.data_string + ''. join(str(i)) + ".csv"
        file = pathlib.Path(Input_string)
        if file.exists():
            indexes.append(i)
    indices = np.array_split(indexes, Ncores)
    return indices


def c_operators(N, Nqubit, H):
    """ Creates the set of collapse operators depending on the problem type
    (MIS/MKC).

    Parameters
    ----------
    N : int
        Number of nodes
    Nqubits: int
        Number of qubits used for the problem

    Returns
    -------
    c_op: list
        List of the collapse operators.
    """
    if parametres.problem == 'MIS':
        c_op = [np.sqrt(parametres.gamma)*qutip.Qobj(
            scipy.sparse.diags(
                qr.sigma_z_operator(H, pos=i), 0))
                for i in range(N)]
    elif parametres.problem == 'MKC':
        c_op = [np.sqrt(parametres.gamma)*qutip.Qobj(
            scipy.sparse.diags(
                qr.sigma_z_operator_total(Nqubit, pos=i), 0))
                for i in range(Nqubit)]
    else:
        raise TypeError("Problem should be MIS or MKC")
    return c_op


main("MIS")
