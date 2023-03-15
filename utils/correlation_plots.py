import torch.nn as nn
import matplotlib.pyplot as plt


def make_correlation_plots(preds, target, layers):
    """
    """
    fig, axs = plt.subplots(2)
    fig.suptitle("Correlation Plots")

    for idx, layer in enumerate(layers):
        axs[idx].plot(preds[idx], target[idx])
        axs[idx].title(layer)
        axs[idx].set_xlabel("Prediction")
        axs[idx].set_ylabel("Target")

    plt.savefig('correlation_plots')
    plt.close()


def make_subplot(fig, axs, idx, layer, type):
    if type == "Linear":
        axs[idx].hist(layer.weight.data.reshape(-1, 1))
        axs[idx].set_title('Sharing Y axis')


def make_weight_hist(model, layers):
    plt.figure()
    fig, axs = plt.subplots(int(len(layers)), 1)
    fig.suptitle('Weight & Bias Distribution')
    for idx, layer in enumerate(layers):
        try:
            layer = getattr(model, layer)
        except Exception as e:
            print(f'Failed retrieving layer {layer}')
            print(e)

        if isinstance(layer, nn.Linear):
            # make_subplot(fig, axs, idx, layer, type="Linear")
            axs[idx].hist(layer.weight.data.reshape(-1, 1))
        elif isinstance(layer, nn.BatchNorm1d):
            make_subplot(fig, axs, idx, layer, type="BatchNorm1d")
        
    plt.savefig('weights.png')
    plt.close()
