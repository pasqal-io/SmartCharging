#!/usr/bin/python3.6
import numpy as np
from os import listdir
import sys

sys.path.append('/home/margarita/these/simulations/POC-Smart-Charging/Experiments/')

from qaoa_maxcut import *


def opt_params_evolution(instances, inits, next_par, p_max):
    n_pars = int((1 + p_max) * p_max / 2)
    opt_betas = np.zeros((len(instances), n_pars))
    opt_gammas = np.zeros((len(instances), n_pars))
    params = inits
    
    for i in range(len(instances)):
            
        for p in range(1, p_max + 1):
            start = int(p * (p-1) / 2)
            opt_betas[i, start:start + p] = params[i][:p]
            opt_gammas[i, start:start + p] = params[i][p:]
            if p != p_max:
                params[i] = next_par(instances[i], params[i])
    return opt_betas, opt_gammas

def app_ratio_evolution(bf_opt, objectives, opt_betas, opt_gammas, p_max): 
    assert (len(bf_opt) == len(objectives))
    
    app_ratios = np.zeros((len(bf_opt), p_max))
    for i in range(len(bf_opt)):
        opt_angles = [opt_betas[i][0], opt_gammas[i][0]]
        app_ratios[i, 0] = objectives[i](opt_angles) 
        
        for p in range(2, p_max+1):           
            start = int(p * (p-1) / 2)
            opt_angles = np.append(opt_betas[i][start:start + p], opt_gammas[i][start:start + p])
            app_ratios[i, p-1] = objectives[i](opt_angles) 
        app_ratios[i] /= bf_opt[i]

    return app_ratios

def main(argv):

    dataset = argv[0]
    dataset_name = argv[1]
    method = argv[2]
    p_max = int(argv[3])
    n_samples = int(argv[4])
    
    logfile = open("../../Results/MaxCut/" + dataset_name + "/" + method + "_log_pmax-" + str(p_max) + "_n-" + str(n_samples) + ".txt", 'w')
    betas_file = "../../Results/MaxCut/" + dataset_name + "/" + method + "_betas_pmax-" + str(p_max) + "_n-" + str(n_samples) + ".csv"
    gammas_file = "../../Results/MaxCut/" + dataset_name + "/" + method + "_gammas_pmax-" + str(p_max) + "_n-" + str(n_samples) + ".csv"
    app_file = "../../Results/MaxCut/" + dataset_name + "/" + method + "_app_ratios_pmax-" + str(p_max) + "_n-" + str(n_samples) + ".csv"

    set_options(logfile, method)

    graphs = [SmartChargingGraph.from_file(dataset + instance) for instance in listdir(dataset)][:n_samples]
    mean_energies = [qaoa_mean_energy(graph) for graph in graphs]
    
    bf_opts = [brute_force_search(graph)for graph in graphs]

    p_1_opt_params = []

    opt_vals = []
    
    for graph in graphs:
        def th_obj(angles):
            return -th_energy_mean(graph, angles[0], angles[1])        
        p_1_opt_params.append(num_opt_par(th_obj, init_pattern(graph.V)))
        opt_vals.append(-th_obj(p_1_opt_params[-1]))
        
    opt_betas, opt_gammas = opt_params_evolution(mean_energies, p_1_opt_params, interp, p_max)
    
    if (graphs[0].V <= 20):
        app_ratios = app_ratio_evolution(bf_opts, mean_energies, opt_betas, opt_gammas, p_max)
    else:
        app_ratios = [ [opt_vals[i] / bf_opts[i]] for i in range(len(opt_vals))]

    np.savetxt(betas_file, opt_betas, delimiter=',')
    np.savetxt(gammas_file, opt_gammas, delimiter=',')
    np.savetxt(app_file, app_ratios, delimiter=',')
    logfile.close()
    
if __name__ == "__main__":
    main(sys.argv[1:])
