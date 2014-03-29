__author__ = 'jakob'

### XOR problem example for ffnet ###

from ffnet import ffnet, mlgraph

# Generate standard layered network architecture and create network


def train(input, target):
    print input[0]
    print len(input)
    conec = mlgraph((16, 16, 10))
    net = ffnet(conec)
    input = input.tolist()
    target = target.tolist()

    print "FINDING STARTING WEIGHTS WITH GENETIC ALGORITHM..."
    # net.train_genetic(input, target, individuals=20, generations=30)
    #then train with scipy tnc optimizer
    print "TRAINING NETWORK..."
    net.train_tnc(input, target, maxfun=1000, messages=1)
    return net


def test(net, input):
    print "TESTING NETWORK..."
    output = net.call(input)
    print output
    return output
