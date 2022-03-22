import sys
import os
import numpy as np
import seaborn
import pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import math

sys.path.insert(0, os.path.split(os.path.realpath(__file__))[0] + "/../Multi-Channel-Time-Encoding/")
from src import *

def error_learned_weights(num_examples, noise_level):
    n_inputs = 20
    n_outputs = 5
    A = np.random.random(size=(n_outputs, n_inputs))
    period = 10
    sampling_time = 10

    signals = []
    spikes_mult = []
    tem_params = TEMParams(1, 1, 1, A)

    np.random.seed(10)
    for n_e in range(num_examples):
        signals.append(Signal.periodicBandlimitedSignals(period))
        for n_i in range(n_inputs):
            signals[-1].add(Signal.periodicBandlimitedSignal(10, 10, np.random.random(size=(10,))))
        spikes = Encoder.ContinuousEncoder(tem_params).encode(signals[-1], sampling_time)
        spikes_mult.append(spikes.corrupt_with_gaussian(noise_level))


    single_layer = Layer(n_inputs, n_outputs)
    single_layer.learn_weight_matrix_from_multi_signals(signals, spikes_mult)

    error = np.linalg.norm(single_layer.weight_matrix-A)
    return error

n_examples = np.arange(1, 15, 1)
n_examples = np.array([1,2,4,8,16,32, 64])
noise_level = np.array([1e-5,3.15e-5, 1e-4, 3.15e-4, 1e-3,3.15e-3, 1e-2,3.15e-2, 1e-1])
noise_level_string = ["1e-05","3.15e-05", "1e-04","3.15e-04", "1e-03","3.15e-03", "1e-02","3.15e-02", "1e-01"]
noise_level_string = ["100dB","90dB", "80dB","70dB", "60dB","50dB", "40dB","30dB", "20dB"]

error = np.zeros((len(n_examples), len(noise_level)))
generate = False
plot = True

if generate:
    for n_e in range(len(n_examples)):
        for n_st in range(len(noise_level)):
            error[n_e, n_st] = error_learned_weights(n_examples[n_e], noise_level[n_st])
        print(n_e)

    data_filename = os.path.split(os.path.realpath(__file__))[0] + "/../Data/single_layer_examples_vs_noise.pkl"
    with open(data_filename, "wb") as f:  # Python 3: open(..., 'wb')
        pickle.dump(
            [n_examples, noise_level, noise_level_string, error], f
         )

if plot:
    plt.rc('text', usetex=False)
    # plt.rc('text.latex', unicode = False)
    plt.rc('svg',fonttype = 'none')
    data_filename = os.path.split(os.path.realpath(__file__))[0] + "/../Data/single_layer_examples_vs_noise.pkl"
    with open(data_filename, "rb") as f:  # Python 3: open(..., 'wb')
        obj = pickle.load(f, encoding="latin1")
    n_examples = obj[0]
    noise_level = obj[1]
    noise_level_string = obj[2]
    noise_level_string = ["100dB", "90dB", "80dB", "70dB", "60dB", "50dB", "40dB", "30dB", "20dB"]
    error = obj[3]
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
    seaborn.heatmap(error[::-1, :], norm=log_norm, yticklabels=n_examples[::-1], xticklabels=noise_level_string,
                    cbar_kws={"ticks": cbar_ticks[1:]}, )
    plt.ylabel("Num Examples")
    plt.xlabel("Noise Level")
    plt.tight_layout()
    plt.savefig(os.path.split(os.path.realpath(__file__))[0] + "/../Figures/single_layer_examples_vs_noise.png")
    plt.savefig(os.path.split(os.path.realpath(__file__))[0] + "/../Figures/single_layer_examples_vs_noise.svg", transparent = True)