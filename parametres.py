def init(experiment, layers=6, n=False):
    global problem
    global data_string
    global result_string

    if experiment == "MIS":
        problem = "MIS"
        data_string = "MIS_data/MIS_data_"
        result_string = "MIS_results/MIS_result_"
    elif experiment == "MKC_P":
        problem = "MKC"
        data_string = "MKC_data/MKC_data_Poisson_"
        result_string = "MKC_results/MKC_Poisson_result_"
    elif experiment == "MKC_U":
        problem = "MKC"
        data_string = "MKC_data/MKC_data_Unif_"
        result_string = "MKC_results/MKC_Unif_result_"
    else:
        raise ValueError("experiment must be of the form MIS, MKC_P or MKC_U")

    global noise
    noise = n

    global l
    l = layers  # number of layers

    global maxfev  # function evaluations authorized per layer
    maxfev = 200

    global gamma  # noise parameter
    gamma = 0.03

    global delta_ee  # Used for the Ising Hamiltonian
    delta_ee = 1.

    global Omega  # Used for the Ising Hamiltonian
    Omega = 1.

    global distance  # Below a given distance, vertices are connected
    distance = 10
