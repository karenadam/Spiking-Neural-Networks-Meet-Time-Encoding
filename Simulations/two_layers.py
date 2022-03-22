import sys
import os
import numpy as np
import seaborn
import pickle
import matplotlib
from matplotlib.colors import LogNorm, Normalize
import matplotlib.pyplot as plt
import math
import itertools

sys.path.insert(0, os.path.split(os.path.realpath(__file__))[0] + "/../Multi-Channel-Time-Encoding/")
from src import *

def error_learned_weights(num_examples, sampling_time, n_spikes_per_hidden_unit):
    n_inputs = 2
    n_hidden = 2
    n_outputs = 4
    W1 = np.random.random(size=(n_hidden, n_inputs))
    W2 = np.random.random(size = (n_outputs, n_hidden))
    period = 10

    signals = []
    spikes_hidden = []
    spikes_output = []
    tem_params_1 = TEMParams(10, 1, 10, W1)
    tem_params_2 = TEMParams(1, 1, 1, W2)

    for n_e in range(num_examples):
        signals.append(Signal.periodicBandlimitedSignals(period))
        for n_i in range(n_inputs):
            signals[-1].add(Signal.periodicBandlimitedSignal(10, 10, np.random.random(size=(10,))))
        spikes_hidden.append(Encoder.ContinuousEncoder(tem_params_1).encode(signals[-1], sampling_time))

        dirac_inputs_next_layer = np.zeros((n_hidden, n_spikes_per_hidden_unit))
        signals_next_layer = Signal.periodicBandlimitedSignals(period)

        for n_h in range(n_hidden):
            dirac_inputs_next_layer[n_h,:] = spikes_hidden[n_e].get_spikes_of(n_h)[:n_spikes_per_hidden_unit]
            fri_signal = src.FRISignal.FRISignal(dirac_inputs_next_layer[n_h,:], np.ones_like(dirac_inputs_next_layer[n_h,:]), period)
            f_s_components = fri_signal.get_fourier_series(np.arange(0, n_spikes_per_hidden_unit*n_hidden, 1).T)
            signals_next_layer.add(Signal.periodicBandlimitedSignal(period, n_spikes_per_hidden_unit*n_hidden, f_s_components))

        spikes_output.append(Encoder.ContinuousEncoder(tem_params_2).encode(signals_next_layer, sampling_time))


    second_layer = Layer(n_hidden, n_outputs, tem_params = tem_params_2)
    layer_1_output_estimated = second_layer.learn_spike_input_and_weight_matrix_from_multi_example(spikes_output, n_spikes_per_hidden_unit*n_hidden, period)

    #find permutation`

    W2_possible_permutations = list(itertools.permutations(np.arange(n_hidden)))
    error_W2 = np.inf
    permutation_chosen_W2 = W2_possible_permutations[0]
    for n_p in range(len(W2_possible_permutations)):
        W2_permuted = W2[:,W2_possible_permutations[n_p]]
        error_permuted = np.linalg.norm(second_layer.weight_matrix-W2_permuted)
        if error_W2>error_permuted:
            error_W2 = error_permuted
            permutation_chosen_W2 = W2_possible_permutations[n_p]


    first_layer = Layer(n_inputs, n_hidden, tem_params = tem_params_1)
    spikes_hidden_estimate = []
    for n_e in range(num_examples):
        spikes_n_h_estimate = spikeTimes(n_hidden)
        for n_h in range(n_hidden):
            spikes_n_h_estimate.add(n_h, layer_1_output_estimated[n_e][n_h])
        spikes_hidden_estimate.append(spikes_n_h_estimate)

    first_layer.learn_weight_matrix_from_multi_signals(signals, spikes_hidden_estimate)


    W1_possible_permutations = list(itertools.permutations(np.arange(n_inputs)))
    error_W1 = np.inf
    permutation_chosen_W1 = W1_possible_permutations[0]
    for n_p in range(len(W1_possible_permutations)):
        # W1_permuted = W1[permutation_chosen_W2,W1_possible_permutations[n_p]]
        W1_permuted = W1[:,W1_possible_permutations[n_p]]
        W1_permuted = W1_permuted[permutation_chosen_W2,:]

        error_permuted = np.linalg.norm(first_layer.weight_matrix-W1_permuted, ord = 1)
        if error_W1>error_permuted:
            error_W1 = error_permuted
            permutation_chosen_W1 = W1_possible_permutations[n_p]



    # error_W2 = min(np.linalg.norm(second_layer.weight_matrix-W2)), np.linalg.norm(second_layer.weight_matrix-W2))
    return error_W1, error_W2

np.random.seed(25)
# er_W1, er_W2 = error_learned_weights(4,10, n_spikes_per_hidden_unit=4)
# print(er_W1, er_W2)

n_examples = np.array([1,2,4,8,16,32])
n_examples = np.array([2]*50)
sampling_time = np.arange(10, 40, 2)
error_W1 = np.zeros((len(n_examples), len(sampling_time)))
error_W2 = np.zeros((len(n_examples), len(sampling_time)))

generate = False
plot = True
if generate:
    for n_e in range(len(n_examples)):
        for n_st in range(len(sampling_time)):
            error_W1[n_e, n_st], error_W2[n_e, n_st] = error_learned_weights(n_examples[n_e], sampling_time[n_st], 4)
            print(error_W1[n_e, n_st], error_W2[n_e, n_st])
        print(n_e)

    data_filename = os.path.split(os.path.realpath(__file__))[0] + "/../Data/two_layers.pkl"
    with open(data_filename, "wb") as f:  # Python 3: open(..., 'wb')
        pickle.dump(
            [n_examples, sampling_time, error_W1, error_W2], f
         )

if plot:
    data_filename = os.path.split(os.path.realpath(__file__))[0] + "/../Data/two_layers.pkl"
    with open(data_filename, "rb") as f:  # Python 3: open(..., 'wb')
        obj = pickle.load(f, encoding="latin1")
    n_examples = obj[0]
    sampling_time = obj[1]
    error_W1 = obj[2]
    error_W2 = obj[3]

    log_norm = LogNorm(vmin=error_W1.min().min(), vmax=error_W1.max().max())
    cbar_ticks = [
        math.pow(10, i)
        for i in range(
            math.floor(math.log10(error_W1.min().min())),
            1 + math.ceil(math.log10(error_W1.max().max())),
        )
    ]
    # seaborn.heatmap(error, norm = log_norm)

    seaborn.heatmap(error_W1[::-1, :], norm = log_norm, yticklabels = n_examples[::-1], xticklabels = sampling_time, cbar_kws = {"ticks": cbar_ticks[1:]},)
    plt.ylabel("Num Examples")
    plt.xlabel("Duration of Exposure")
    plt.savefig(os.path.split(os.path.realpath(__file__))[0] + "/../Figures/two_layers_W1.png")

    plt.figure()
    log_norm = LogNorm(vmin=error_W2.min().min(), vmax=error_W2.max().max())
    cbar_ticks = [
        math.pow(10, i)
        for i in range(
            math.floor(math.log10(error_W2.min().min())),
            1 + math.ceil(math.log10(error_W2.max().max())),
        )
    ]
    # seaborn.heatmap(error, norm = log_norm)

    seaborn.heatmap(error_W2[::-1, :], norm = log_norm, yticklabels = n_examples[::-1], xticklabels = sampling_time, cbar_kws = {"ticks": cbar_ticks[1:]},)
    plt.ylabel("Num Examples")
    plt.xlabel("Duration of Exposure")
    plt.savefig(os.path.split(os.path.realpath(__file__))[0] + "/../Figures/two_layers_W2.png")

    clr = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    plt.figure(figsize = (5,3))
    plt.rc('text', usetex=False)
    # plt.rc('text.latex', unicode = False)
    plt.rc('svg',fonttype = 'none')
    plt.plot(sampling_time, np.median(error_W1,0), color = clr[0], label = 'W1')
    plt.plot(sampling_time, np.quantile(error_W1, 0.25, 0), color = clr[0], alpha = 0.3, linestyle = '--')
    plt.plot(sampling_time, np.quantile(error_W1, 0.75, 0), color = clr[0], alpha = 0.3, linestyle = '--')
    plt.gca().fill_between(sampling_time, np.quantile(error_W1, 0.25, 0),np.quantile(error_W1, 0.75, 0), color=clr[0], alpha=0.1)

    plt.plot(sampling_time, np.median(error_W2,0), color = clr[1],label = 'W2')
    plt.plot(sampling_time, np.quantile(error_W2, 0.25, 0), color = clr[1], alpha = 0.3, linestyle = '--')
    plt.plot(sampling_time, np.quantile(error_W2, 0.75, 0), color = clr[1], alpha = 0.3, linestyle = '--')
    plt.gca().fill_between(sampling_time, np.quantile(error_W2, 0.25, 0),np.quantile(error_W2, 0.75, 0), color=clr[1], alpha=0.1)
    plt.gca().set_yscale('log')
    plt.legend(loc = 'best')
    plt.ylabel("Reconstruction Error")
    plt.xlabel("Duration of Exposure")
    plt.title("Reconstruction Error on Weight Matrices")
    plt.tight_layout()
    plt.savefig(os.path.split(os.path.realpath(__file__))[0] + "/../Figures/two_layers_plot.png")
    plt.savefig(os.path.split(os.path.realpath(__file__))[0] + "/../Figures/two_layers_plot.svg", transparent = True)