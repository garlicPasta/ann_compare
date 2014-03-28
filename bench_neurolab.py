import numpy as np
import neurolab as nl


def train(input, target):
    n_samples = len(input)
    n_input_neurons = len(input[1])
    n_output_neurons = len(target[1])
    net = nl.net.newff([[1.0, 0] for _ in range(n_input_neurons)], [16, n_output_neurons])
    net.trainf = [nl.net.trans.TanSig, nl.net.trans.TanSig, nl.net.trans.TanSig, nl.net.trans.TanSig, nl.net.trans.SoftMax]
    # Train process
    net.trainf = nl.net.train.train_rprop
    net
    err = net.train(input, target, goal=700, show=1)
    return net


def test(net, input):
    return net.sim(input)

