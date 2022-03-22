import sys
import os
import numpy as np
import seaborn
import pickle
import matplotlib
from matplotlib.colors import LogNorm, Normalize
import matplotlib.pyplot as plt
import math

sys.path.insert(0, os.path.split(os.path.realpath(__file__))[0] + "/../Multi-Channel-Time-Encoding/")
from src import *

def error_learned_weights(num_examples, sampling_time):
    n_inputs = 20
    n_outputs = 5
    A = np.random.random(size=(n_outputs, n_inputs))
    period = 10

    signals = []
    spikes_mult = []
    tem_params = TEMParams(1, 1, 1, A)

    np.random.seed(10)
    for n_e in range(num_examples):
        signals.append(Signal.periodicBandlimitedSignals(period))
        for n_i in range(n_inputs):
            signals[-1].add(Signal.periodicBandlimitedSignal(10, 10, np.random.random(size=(10,))))
        spikes_mult.append(Encoder.ContinuousEncoder(tem_params).encode(signals[-1], sampling_time))


    single_layer = Layer(n_inputs, n_outputs)
    single_layer.learn_weight_matrix_from_multi_signals(signals, spikes_mult)

    error = np.linalg.norm(single_layer.weight_matrix-A)
    return error

n_examples = np.arange(1, 10, 1)
sampling_time = np.arange(1, 15, 1)
error = np.zeros((len(n_examples), len(sampling_time)))
generate = False
plot = True

if generate:
    for n_e in range(len(n_examples)):
        for n_st in range(len(sampling_time)):
            error[n_e, n_st] = error_learned_weights(n_examples[n_e], sampling_time[n_st])
        print(n_e)

    data_filename = os.path.split(os.path.realpath(__file__))[0] + "/../Data/single_layer_spikes_vs_examples.pkl"
    with open(data_filename, "wb") as f:  # Python 3: open(..., 'wb')
        pickle.dump(
            [n_examples, sampling_time, error], f
         )

if plot:
    plt.rc('text', usetex=False)
    # plt.rc('text.latex', unicode = False)
    plt.rc('svg',fonttype = 'none')
    data_filename = os.path.split(os.path.realpath(__file__))[0] + "/../Data/single_layer_spikes_vs_examples.pkl"
    with open(data_filename, "rb") as f:  # Python 3: open(..., 'wb')
        obj = pickle.load(f, encoding="latin1")
    n_examples = obj[0]
    sampling_time = obj[1]
    error = obj[2]

    log_norm = LogNorm(vmin=error.min().min(), vmax=error.max().max())
    cbar_ticks = [
        math.pow(10, i)
        for i in range(
            math.floor(math.log10(error.min().min())),
            1 + math.ceil(math.log10(error.max().max())),
        )
    ]
    # seaborn.heatmap(error, norm = log_norm)
    plt.figure(figsize = (5,3))

    seaborn.heatmap(error[::-1, :], norm = log_norm, yticklabels = n_examples[::-1], xticklabels = sampling_time, cbar_kws = {"ticks": cbar_ticks[1:]},)
    plt.ylabel("Num Examples")
    plt.xlabel("Duration of Exposure")
    plt.tight_layout()
    plt.savefig(os.path.split(os.path.realpath(__file__))[0] + "/../Figures/single_layer_spikes_vs_examples.png")
    plt.savefig(os.path.split(os.path.realpath(__file__))[0] + "/../Figures/single_layer_spikes_vs_examples.svg", transparent = True)