import glob
import time
import copy
import os
import numpy as np


def translate_digits(digits):
    zeros = np.array([0] * 10)
    target = []
    for d in digits:
        z = copy.copy(zeros)
        z[d] = 1
        target.append(z)
    return np.array(target)


def load(filename):
    data = np.genfromtxt(filename)
    input = data[:, 0:-1]
    return input / 100.0, translate_digits(data[:,-1])


def load_benchmarks(network_files=""):
    print network_files
    if network_files == "":
        benchmarks = glob.glob("bench_*.py")
    else:
        benchmarks = glob.glob(network_files)
    bench_modules = []
    pwd = os.path.dirname(__file__)
    for bench in benchmarks:
        name = os.path.basename(bench).rstrip(".py")
        bench_modules.append(__import__(name))
    return bench_modules


def pick_best(output):
    max_i = np.argmax(output, 1)
    return translate_digits(max_i)


def evaluation(before_train, after_train, after_test, output, expected):
    train_spend = after_train - before_train
    test_spend = after_test - after_train
    print("Time for training: {}".format(train_spend))
    print("Time for testing: {}".format(test_spend))
    best_output = pick_best(output)
    n_of_samples = len(output)
    same = np.zeros((n_of_samples, 1))
    for i in xrange(n_of_samples):
        same[i] = np.alltrue(best_output[i, :] == expected[i, :])

    right = np.sum(same)
    print("Accuracy: {}".format(right / n_of_samples))


def run_benchmark():
    # if set only specfic file gets tested
    specific_network ="bench_ffnet.py"

    # Load testingset
    train_input, train_target = load('pendigits-training.txt')
    test_input, test_target = load('pendigits-testing.txt')

    if 'specific_network' in locals():
        benchmarks = load_benchmarks(specific_network)
    else :
        benchmarks = load_benchmarks()

    for benchmark in benchmarks:
        before_train = time.time()
        net = benchmark.train(train_input, train_target)
        after_train = time.time()
        predicted = benchmark.test(net, test_input)
        after_test = time.time()
        evaluation(before_train, after_train, after_test, predicted, test_target)

if __name__ == '__main__':
    run_benchmark()
