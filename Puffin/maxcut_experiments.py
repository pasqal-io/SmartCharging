from os import listdir
from qaoa_maxcut import *

def read_instance(filename):
    data_file = open(filename, "r")
    jobs = data_file.readlines()[1:]
    durations, priorities = zip(*[tuple(map(int, job.split('\t'))) for job in jobs])
    return (durations, priorities)

def bf_opt_par(objective, n_steps=100, b_max=0.35, g_max=0.04, b_min=0, g_min=0):
    betas = np.linspace(b_min, b_max, n_steps)
    gammas = np.linspace(g_min, g_max, n_steps)
    X, Y = np.meshgrid(betas, gammas)
    means = objective(betas, gammas)
    max_ind = np.unravel_index(np.argmax(means, axis=None), means.shape)
    
    return [X.T[max_ind], Y.T[max_ind]]

def qaoa_1(instance):

    t_list, p_list = instance
    graph = SmartChargingGraph(p_list, t_list)

    def f(angles):
        return -th_energy_mean(graph, angles[0], angles[1])

    opt_param = num_opt_par(f, init_pattern(graph.V))
    mean_energy = -f(opt_param)
    return wmft_from_cut(instance, mean_energy), opt_param

"""Specify the input dataset"""

def main():

    dir_prefix = "puffin_data/MC_data/BelibMai2017"

    dataset_suffix = {6:".6ChargesP_1_2_3/", 8:".8ChargesP_1_3_4/", 10:".10ChargesP_2_3_5/", 15:".15ChargesP_2_3_10/", 30:".30ChargesP_4_6_20/" ,50:".50ChargesP_25_50_75/", 70:".70ChargesP_12_23_35/", 100:".100ChargesP_17_33_50/", 150:".150ChargesP_8_17_25/", 170:".170ChargesP_29_56_85/", 200:".200ChargesP_34_66_100/", 220:".220ChargesP_37_73_110/", 250:".250ChargesP_43_82_125/"}

    instances = {}
    for ds in dataset_suffix.items():
        instances[ds[0]] = []
        for instance_file in listdir(dir_prefix + ds[1]):
            instance = read_instance(dir_prefix + ds[1] + instance_file)
            if instance[1][0] <= 100 and 0 < instance[1][0]:
                instances[ds[0]].append(instance)

    rand_sols = {}
    dp_sols = {}
    qaoa_sols = {}
    betas = {}
    gammas = {}

    for ds in instances.items():

        rand_sols[ds[0]] = []
        dp_sols[ds[0]] = []
        qaoa_sols[ds[0]] = []
        betas[ds[0]] = []
        gammas[ds[0]] = []

        for instance in ds[1][:n_max]:
            rand_sols[ds[0]].append(rand_expectation(instance))
            dp_sols[ds[0]].append(dp_exact(instance))
            qaoa_sol, qaoa_par = qaoa_1(instance)
            qaoa_sols[ds[0]].append(qaoa_sol)
            betas[ds[0]].append(qaoa_par[0])
            gammas[ds[0]].append(qaoa_par[1])

if __name__ == "__main__":
    main()