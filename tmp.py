from utils import get_network

from extra_models import *

import torch
import netron, torchsummary

def visualize_network(model_name):
    # plot neuxon network structure
    net = get_network(model_name, channel=3, num_classes=10, dist=False) # dist=False to use the original model instead of DataParallel model

    '''print summary'''
    net.cuda()
    torchsummary.summary(net, (3, 32, 32))

    '''Visualize the network structure'''
    net.cpu()
    input_ = torch.randn(1, 3, 32, 32)
    output = net(input_)

    onnx_path = model_name + ".onnx"
    torch.onnx.export(net, input_, onnx_path)

    netron.start(onnx_path)

if __name__ == "__main__":
    # visualize_network('ConvNet')
    visualize_network('ResNet18')
    # visualize_network('ResNet20_CIFAR') # BUG
