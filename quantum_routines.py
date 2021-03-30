import numpy as np
import igraph
import csv
import copy
import qutip
from scipy.optimize import differential_evolution
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix

import parametres


def quantum_loop(psi, H_0, H_c, param):
    """Executes the quantum loop of QAOA.

    Parameters
    ----------
    psi : qutip.Qobj()
        Initial quantum state.
    H_0 : qutip.Qobj()
        Miximing Hamiltonian
    H_c : qutip.Qobj()
        Cost Hamiltonian
    param: array
        set of angles/times that parametrize each layer of QAOA.
        For p = 2 layers, the array is of the form [t_1,t_2,tau_1,tau_2].

    Returns
    -------
    psi : Qobj()
        The final state, after evolution of initial state under all layers of
        the QAOA.

    """
    middle = int(len(param)/2)
    t = param[:middle]
    tau = param[middle:]
    p = len(t)
    if parametres.noise:
        if parametres.problem == 'MKC':
            for i in range(p):
                psi = time_evolution_mc(psi, H_c, t[i])
                psi = time_evolution_mc(psi, H_0, tau[i])
            return psi
        elif parametres.problem == 'MIS':
            for i in range(p):
                psi = time_evolution_mc(psi, H_0, tau[i])
                psi = time_evolution_mc(psi, H_0 + H_c, t[i])
                psi = time_evolution(psi, H_0, tau[-1])
            return psi
        else:
            raise ValueError("Problem type must be 'MIS' or 'MKC'")
    elif not parametres.noise:
        if parametres.problem == 'MKC':
            for i in range(p):
                psi = time_evolution(psi, H_c, t[i])
                psi = time_evolution(psi, H_0, tau[i])
            return psi
        elif parametres.problem == 'MIS':
            for i in range(p):
                psi = time_evolution(psi, H_0, tau[i])
                psi = time_evolution(psi, H_0 + H_c, t[i])
                psi = time_evolution(psi, H_0, tau[-1])
            return psi
        else:
            raise ValueError("Problem type must be 'MIS' or 'MKC'")
    else:
        raise ValueError('noise must be a Boolean')


def time_evolution(psi, H, t):
    """
    Schrodinger evolution of a state psi under a Hamiltonian H, during
    a time t.

    Parameters
    ----------
    psi: qutip.Qobj()
        quantum state under evolution
    H: qutip.Qobj()
        Hamiltonian
    t: scalar
        time during which H is applied to psi

    Returns
    -------
    psi2 : qutip.Qobj()
        final state
    """
    if t == 0:
        return psi
    else:
        tlist = np.linspace(0, t, 20)
        psi2 = qutip.sesolve(H, psi, tlist).states[-1]
    return psi2


def time_evolution_mc(psi, H, t, c_op):
    """ Monte-Carlo evolution in the case of a noisy system.

    Parameters
    ----------
    psi: qutip.Qobj()
        quantum state under evolution
    H: qutip.Qobj()
        Hamiltonian
    t: scalar
        time during which H is applied to psi
    c_op: list
        single collapse operator or a list of collapse operators.

    Returns
    ------
    psi2: qutip.Qobj()
        final state
    """
    tlist = np.linspace(0, t, 50)
    psi2 = qutip.mcsolve(H, psi, tlist, c_ops=c_op, ntraj=100,
                         progress_bar=None).states[0][-1]
    return psi2


def func(param, *args):
    """ Function optimized by QAOA. Returns the expectation value of psi_final,
    given parameters [t,tau].

    Parameters
    ----------
    param: array
        set of angles/times that parametrize each layer of QAOA.
        For p = 2 layers, the array is of the form [t_1,t_2,tau_1,tau_2].
    *args: tuple
        tuple of all arguments necessary to evaluate func

    Returns
    -------
    cost(H_c,psi) : scalar
                    Expectation value <psi|H_c|psi>
    """
    psi_0, H_0, H_c, H_c_plus = args
    psi = quantum_loop(psi_0, H_0, H_c_plus, param)
    return qutip.expect(H_c, psi)


def new_population(x):   # x is the best result found at previous layer,
    """Returns an initial population for layer l+1, an educated guess given the
    optimal parameters at layer l.
    The phase space explored depends on the problem type (MIS/MKC).

    Parameters
    ----------
    x: array
        Optimal parameters found at the previous layer. Dimension is 2*l, where
        l is the number of layers at the previous round.

    Returns
    -------
    new_pop: array
        Initial population for layer l+1. Dimension is 2*(l+1). """
    middle = len(x)//2
    new_pop = []  # np.zeros((25, len(x)+2))
    t = x[:middle]
    tau = x[middle:]
    if parametres.problem == 'MKC':
        new_t = np.linspace(-2*np.pi, 2*np.pi, endpoint=True, num=5)
        new_tau = np.linspace(-2*np.pi, 2*np.pi, endpoint=True, num=5)
    if parametres.problem == 'MIS':
        new_t = np.linspace(-np.pi/4, np.pi/4, endpoint=True, num=5)
        new_tau = np.linspace(-np.pi/2, np.pi/2, endpoint=True, num=5)
    X, Y = np.meshgrid(new_t, new_tau)
    for i in range(len(X)):
        for j in range(len(Y)):
            indiv = np.concatenate((
                np.concatenate((t, [X[i][j]])),
                np.concatenate((tau, [Y[i][j]])))).tolist()
            new_pop.append(indiv)
    return np.asarray(new_pop)


def param_evolution_with_layers(func, maxfev, l, *argus):
    # Start by finding the minima for first layer

    """ Heart of the QAOA evolution with layers. Adds layer-by-layer in order
    to find the optimal parameters for the wanted number of layers l.

    Parameters
    ----------
    func : callable
        The objective function to be minimized.
    argus: tuple
        Tuple of any additional fixed parameters needed to
        completely specify the function.
    maxfev : int
        Maximum number of func evalution allowed. This is done in order to
        restrain the budget of function calls (expensive)
    l : int
        Number of layers in QAOA

    Returns
    -------
    param: array
        set of optimal angles/times that parametrize each layer of QAOA for
        best result.
    cost: array
        Best approximation ratio achieved for each layer. dimension of the
        array is l.
     """
    psi_0, H_0, H_c, H_c_plus = argus
    cost = []
    param = []
    if parametres.problem == 'MKC':
        bounds = [(-2*np.pi, 2*np.pi), (-2*np.pi, 2*np.pi)]
        maxit = int(maxfev/(5*2) - 1)
        print('layer 1 beginning')
        res = differential_evolution(func, bounds,
                                     args=argus,
                                     strategy='best1bin',
                                     maxiter=maxit*5,
                                     popsize=5,
                                     tol=0.00001,
                                     mutation=1.6,
                                     recombination=0.7,
                                     seed=None,
                                     callback=None,
                                     disp=False,
                                     polish=True,
                                     init='latinhypercube',
                                     atol=0,
                                     updating='deferred',
                                     workers=-1)
        x = res.x
        print('approximation ratio achieved:', abs(res.fun/np.min(H_c)))
        param.append(x.tolist())
        cost.append(np.real(res.fun/np.min(H_c)))

        # educated guess for next layers

        for i in range(l-1):
            nou_pop = new_population(x)
            A = [(0.0001, np.pi) for i in range(len(nou_pop[0])//2)]
            B = [(0.0001, np.pi/2) for i in range(len(nou_pop[0])//2)]
            bounds = np.concatenate((A, B))
            print(i+2, 'layers beginning')
            res = differential_evolution(func,
                                         bounds,
                                         args=argus,
                                         strategy='best1bin',
                                         maxiter=maxit,
                                         popsize=5,
                                         tol=0.00001,
                                         mutation=1.6,
                                         recombination=0.7,
                                         seed=None,
                                         callback=None,
                                         disp=False,
                                         polish=True,
                                         init=nou_pop,
                                         atol=0,
                                         updating='deferred',
                                         workers=-1)
            x = res.x
            print('approximation ratio achieved:', abs(res.fun/np.min(H_c)))
            param.append(x.tolist())
            cost.append(np.real(abs(res.fun/np.min(H_c))))

    if parametres.problem == 'MIS':
        bounds = [(-np.pi/2, np.pi/2),
                  (-np.pi/4, np.pi/4),
                  (-np.pi/4, np.pi/4)]
        maxit = int(maxfev/(5*2) - 1)
        print('layer 1 beginning')
        res = differential_evolution(func,
                                     bounds,
                                     args=argus,
                                     strategy='best1bin',
                                     maxiter=5*maxit,
                                     popsize=5,
                                     tol=0.00001,
                                     mutation=(0.5, 1.2),
                                     recombination=0.7,
                                     seed=None,
                                     callback=None,
                                     disp=False,
                                     polish=True,
                                     init='latinhypercube',
                                     atol=0,
                                     updating='deferred',
                                     workers=-1)
        x = res.x
        psi = quantum_loop(psi_0, H_0, H_c_plus, x)
        mod_cost = qutip.expect(H_c, psi)
        print('approximation ratio achieved:', abs(res.fun/np.min(H_c)))
        param.append(x.tolist())
        cost.append(np.real(abs(mod_cost/np.min(H_c))))

        # educated guess for next layers
        for i in range(l-1):
            nou_pop = new_population(x)
            f = len(nou_pop[0])
            A = [(-2*np.pi, 2*np.pi) for i in range(f//2)]
            B = [(-2*np.pi, 2*np.pi) for i in range(f - f//2)]
            bounds = np.concatenate((A, B))
            print(i+2, 'layers beginning')
            res = differential_evolution(func,
                                         bounds,
                                         args=argus,
                                         strategy='best1bin',
                                         maxiter=5*maxit,
                                         popsize=5,
                                         tol=0.00001,
                                         mutation=(0.5, 1.2),
                                         recombination=0.7,
                                         seed=None,
                                         callback=None,
                                         disp=False,
                                         polish=True,
                                         init=nou_pop,
                                         atol=0,
                                         updating='deferred',
                                         workers=-1)
            x = res.x
            func_res = res.fun
            for ii in range(5):
                if ((abs(func_res/np.min(H_c)) <= cost[-1])
                        or (abs(x[:f//2][-1]) < 1e-2 and abs(x[-1] < 1e-2))):
                    print("starting layer again..")
                    res = differential_evolution(func,
                                                 bounds,
                                                 args=argus,
                                                 strategy='best1bin',
                                                 maxiter=5*maxit,
                                                 popsize=5,
                                                 tol=0.00001,
                                                 mutation=(0.5, 1.2),
                                                 recombination=0.7,
                                                 seed=None,
                                                 callback=None,
                                                 disp=False,
                                                 polish=True,
                                                 init=nou_pop,
                                                 atol=0,
                                                 updating='deferred',
                                                 workers=-1)
                    x = res.x
                    func_res = res.fun
                else:
                    break
            psi = quantum_loop(psi_0, H_0, H_c_plus, x)
            mod_cost = qutip.expect(H_c, psi)
            print('approximation ratio achieved:', abs(res.fun/np.min(H_c)))
            param.append(x.tolist())
            cost.append(np.real(abs(mod_cost/np.min(H_c))))
    return param, cost


def data_to_adj_matrix(String):
    """"Transforms .csv raw data files into the adjacency matrix of the graph
    (MKC only)

    Parameters
    ----------
    String: .csv file
            Raw data of the MKC problem. Is usually written as
            (duration,priority index) per line.

    Returns
    -------
    A : array(N,N)
        Adjacency matrix of the graph with N nodes.
    """
    dic = {}
    i = 0
    with open(String, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            for row in csv_reader:
                dic[i] = (int(row[0]), int(row[1]))
                # row[0] = charging time (w) , row[1] = priority index (p)
                i += 1
    # transform dictionary to adjacency matrix
    A = np.zeros((len(dic), len(dic)), dtype=int)
    for i in range(len(A)):
        for j in range(i+1, len(A)):
            A[i][j] = A[j][i] = int(min(dic[i][0]*dic[j][1],
                                        dic[j][0]*dic[i][1])/100)
    return A


def cost_hamiltonian(A):
    """ Returns the cost Hamiltonian from the adjacency matrix of the graph.

        Parameters
        ----------
        A : array(N,N)
            Adjacency matrix of the graph with N nodes

        Returns
        -------
        H : array(2**Nqubit,2**Nqubit)
            Cost Hamiltonian

    """
    N = len(A)
    Nqubit = N*2
    H = np.zeros(2**Nqubit, dtype=complex)
    for k in range(2**Nqubit):
        long_string = format(k, '0'+format(Nqubit, 'd')+'b')
        k_list = list(map(int, str(long_string)))
        k_odd = []
        k_even = []
        for s in range(Nqubit):
            if s % 2 == 0:
                k_even.append(k_list[s])
            else:
                k_odd.append(k_list[s])
        for i in range(N):

            for j in range(i, N):

                if k_even[i] != k_even[j]:
                    H[k] -= A[i][j]/2

                if k_odd[i] != k_odd[j]:
                    H[k] -= A[i][j]/2

                if (k_even[i] + k_even[j] + k_odd[i] + k_odd[j]) % 2 != 0:
                    H[k] -= A[i][j]/2
    return H


def H_ref(A):
    """ Returns the mixing Hamiltonian from the adjacency matrix of the graph.

        Parameters
        ----------
        A : array(N,N)
            Adjacency matrix of the graph with N nodes

        Returns
        -------
        H : scipy.sparse.csc_matrix
            Mixing Hamiltonian used in QAOA

    """
    row = []
    col = []
    N = len(A)
    Nqubit = N*2
    indexes = [np.asarray(
                list(map(int, str(format(i, '0'+format(Nqubit, 'd')+'b')))))
               for i in range(2**Nqubit)]
    indexes_list = [list(
        map(int, str(format(i, '0'+format(Nqubit, 'd')+'b'))))
                    for i in range(2**Nqubit)]
    for i in range(len(indexes)):
        indiv = indexes[i]
        for m in range(len(indiv)):
            indiv_2 = indiv.copy()
            if indiv[m] == 0:
                indiv_2[m] = 1
                # ne pas changer un 1--> 0 car sinon l'index j < i, or H[i][j]
                # deja trouvé quand j'explorais j donc inutile
                # trouver l'index j auquel correspond indiv_2 et implémenter
                # H[i][j] = 1
                j = indexes_list.index(indiv_2.tolist())
                # if j > i : #si j < i, j'ai
                row.append(i)
                col.append(j)
                row.append(j)
                col.append(i)
    data = [1+0*1j for i in range(len(col))]
    H_0_sparse = csc_matrix(coo_matrix(
        (data, (row, col)), shape=(2**Nqubit, 2**Nqubit)))
    return H_0_sparse


def generate_graph(pos):
    """(MIS)  Given the positions of the atoms, build the connectivity graph.

    Parameters
    ----------
    pos: list of lists
        A list per node, where each sublist [x,y] are the coordinates of
        the node in the 2D plane

    Returns
    -------
    g: igraph.Graph()
        Connectivity graph of the atoms
    """
    edges = []
    for n in range(len(pos)-1):
        for m in range(n+1, len(pos)):
            dist = ((pos[m][0]-pos[n][0])**2+(pos[m][1]-pos[n][1])**2)**0.5
            if dist < parametres.distance:
                edges.append([n, m])
    g = igraph.Graph()
    g.add_vertices(len(pos))
    g.add_edges(edges)
    return g


def get_indices(basis_vector_loc, indices):
    """
    This function will return the indices for which the basis vectors are
    possibly connected to the input vector by a sigma^x operator.
    Increasing number of excitations.
    """
    n_initial = indices[len(basis_vector_loc)+1]
    if not len(basis_vector_loc)+2 < len(indices):
        return (-1, -1)
    n_final = indices[len(basis_vector_loc)+2]
    return (n_initial, n_final)


def indices_n_exc_basis_states(H):
    """
    This function returns the indices such that list_indices_n_exc[n_0]
    is the smallest index jj for which the vector jj has n_0 excitations.
    """
    list_indices_n_exc = [0]
    nn = 0
    for mm in range(len(H)):
        if len(H[mm]) > nn:
            list_indices_n_exc.append(mm)
            nn += 1
    list_indices_n_exc.append(len(H))
    return list_indices_n_exc


def generate_Hilbert_space(graph):
    """
    This function generates the Hilbert space, either from a igraph object
    of from a Graph object as defined above.
    """
    H = graph.independent_vertex_sets(min=0, max=0)
    H.insert(0, ())
    indices_nexc_H = indices_n_exc_basis_states(H)
    return (H, indices_nexc_H)


def sigma_x_operator(basis_vector, indices, pos_sigma=-1):
    """Operator that creates the matrix representation of sigma_x."""
    M = sigma_moins_operator(basis_vector, indices, pos_sigma=-1)
    return M+np.transpose(M)


def sigma_moins_operator(basis_vector, indices, pos_sigma):
    """
    Operator that creates the matrix representation of sigma_+
        we create an overloading variable pos_sigma, that denotes the
        position of the sigma_+
        if pos_sigma=-1, global operation.
    """
    dim = len(basis_vector)
    sigma_x_matrix = np.zeros((dim, dim))
    for ii in range(dim-1):
        basis_vector_ii = basis_vector[ii]
        (n_initial, n_final) = get_indices(basis_vector[ii], indices)
        if n_initial < 0. or n_final < 0.:
            continue
        for jj in range(n_initial, n_final):
            basis_vector_jj = basis_vector[jj]
            if pos_sigma > -0.1:
                loc1 = list(copy.copy(basis_vector_ii))
                loc1.append(pos_sigma)
                if set(loc1) == set(basis_vector_jj):
                    sigma_x_matrix[ii, jj] = 1.
                    continue
            else:
                if(set(basis_vector_ii).issubset(set(basis_vector_jj))):
                    sigma_x_matrix[ii, jj] = 1.
    return sigma_x_matrix


def sigma_z_operator(basis_vector, pos=-1):
    """
    Operator that creates the matrix representation of sigma_z.
    As sigma^z is diagonal in the computational basis,
    we will only return a vector-type array and later apply element-wise
    multiplication with the wavefunction
    if pos=-1, global operation.
    """
    dim = len(basis_vector)
    sigma_z_matrix = np.zeros(dim)

    if pos > -0.1:
        for jj in range(dim):
            if (set([pos]).issubset(set(basis_vector[jj]))):
                sigma_z_matrix[jj] = 1.

    # Global operator, all positions
    else:
        for jj in range(dim):
            leng = len(basis_vector[jj])
            sigma_z_matrix[jj] = leng

    return sigma_z_matrix


def sigma_z_operator_total(Nqubit, pos):
    """
    Operator that creates the matrix representation of sigma_z.
    As sigma^z is diagonal in the computational basis,
    we will only return a vector-type array and later apply element-wise
    multiplication with the wavefunction.
    if pos=-1, global operation.
    """
    dim = 2**Nqubit
    sigma_z_matrix = np.zeros(dim)
    for k in range(dim):
        long_string = format(k, '0'+format(Nqubit, 'd')+'b')
        k_list = list(map(int, str(long_string)))
        if k_list[pos] == 0:
            sigma_z_matrix[k] = 1
    return sigma_z_matrix


def generate_Hamiltonians(Hilbert_space, indices_coupling):
    """
    This function returns the Hamiltonians of QAOA.
    The first one is the mixing Hamiltonian.
    The second-one is the phase-separation (cost) Hamiltonian.
    The third one is the dissipation.
    """
    H_Rabi = parametres.Omega*sigma_x_operator(Hilbert_space, indices_coupling)
    H_detuning = parametres.delta_ee*sigma_z_operator(Hilbert_space)
    H_dissipation = 0.

    return (csr_matrix(H_Rabi), H_detuning, H_dissipation)


def sol_to_pos(String):
    """ Converts raw data to position list of lists

    Parameters
    -----------
    String : .csv file
        Raw data of the MIS problem. Is usually written as (x,y)-position per
        line.

    Returns
    -------
    pos: list of lists
        A list per node, where each sublist [x,y] are the coordinates of the
        node in the 2D plane.
    """

    pos = []
    with open(String, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            for row in csv_reader:
                posi = []
                posi.append(int(row[1]))
                posi.append(int(row[2]))
                pos.append(posi)
    return pos
