import os
from tqdm import tqdm
from copy import deepcopy
import json
import time

import numpy as np
import matplotlib.pyplot as plt

from utils import mini_batch_gradient_descent, predict_label
from modules import Linear, ReLU, SoftmaxCrossEntropy
from dataloader import DataLoader


def create_model(num_hidden_layers: int, input_dim: int, output_dim: int, hidden_dim: int) -> dict:
    """Create a neural network model.

    """
    model = {'num_hidden_layers': num_hidden_layers}
    dim_list = [input_dim] + [hidden_dim] * num_hidden_layers
    # print(f'dim_list: {dim_list}')

    for i in range(num_hidden_layers):
        model[f'linear{i}'] = Linear(dim_list[i], dim_list[i + 1])
        model[f'activation{i}'] = ReLU()
    model[f'linear{num_hidden_layers}'] = Linear(hidden_dim, output_dim)
    model['loss'] = SoftmaxCrossEntropy()

    return model


def forward_pass(model: dict, x, y=None, calc_loss: bool = True):
    """Forward pass of the neural network model.

    """
    cache = []
    # print(x[0], x[1], x[2])
    for i in range(model['num_hidden_layers']):
        # print(x.shape)
        cache.append(x)
        x = model[f'linear{i}'].forward(x)
        # print(x.shape)
        cache.append(x)
        x = model[f'activation{i}'].forward(x)

    # print(x.shape)
    cache.append(x)
    o = model[f'linear{model["num_hidden_layers"]}'].forward(x)
    # print(o.shape)
    cache.append(o)
    if calc_loss:
        loss = model['loss'].forward(o, y)
        # print(loss.shape)
        cache.append(loss)
        # print(o[0][:5], o[1][:5])
        # print(o.shape)

    return cache


def backward_pass(model: dict, cache: list):
    """Backward pass of the neural network model.

    """
    num_hidden_layers = model['num_hidden_layers']
    grad = model['loss'].backward(cache[-2], cache[-1])
    # print(grad.shape)
    grad = model[f'linear{num_hidden_layers}'].backward(cache[-3], grad)
    # print(grad.shape)
    for i in range(num_hidden_layers):
        grad = model[f'activation{num_hidden_layers - i - 1}'].backward(cache[-4 - i * 2], grad)
        # print(grad.shape)
        grad = model[f'linear{num_hidden_layers - i - 1}'].backward(cache[-4 - i * 2 - 1], grad)
        # print(grad.shape)


def compute_accuracy_and_loss(data_loader: DataLoader, model: dict, minibatch_size: int = 1000):
    """Compute the accuracy and the average cross-entropy loss of the neural network model.

    """
    acc = 0
    cnt = 0
    loss_avg = 0.0
    num_data = len(data_loader)
    assert num_data >= minibatch_size, (f"Minibatch size {minibatch_size} must be smaller "
                                        f"than or equal to the number of data points {num_data}.")

    for i in range(int(np.floor(num_data / minibatch_size))):
        x, y = data_loader.sample(np.arange(i * minibatch_size, (i + 1) * minibatch_size))

        cache = forward_pass(model, x, y)
        # print(f"cache[-2]: {cache[-2].shape}")
        loss_avg += cache[-1]
        preds = predict_label(cache[-2])
        # print(f"preds: {preds.shape}, y: {y.shape}")
        acc += np.sum(preds == y)
        cnt += 1

    # print(f"acc: {acc}, cnt: {cnt}, minibatch size: {minibatch_size}")
    return acc / (cnt * minibatch_size), loss_avg / cnt


def train_and_eval(config: dict, evaluate: bool = True):
    # Set random seed
    seed = config.get('random_seed', None)
    np.random.seed(seed)

    # Prepare dataset
    train_data_file = config['train_data_file']
    train_label_file = config['train_label_file']
    test_data_file = config.get('test_data_file', '')
    test_label_file = config.get('test_label_file', '')

    train_set = DataLoader(train_data_file, train_label_file)
    train_max_min_price = (train_set.max_price, train_set.min_price)
    mu_std_price = (train_set.mu_price, train_set.std_price)
    if evaluate:
        test_set = DataLoader(test_data_file, test_label_file, mu_std_price=mu_std_price)
    n_train = len(train_set)

    # Build MLP
    num_hidden_layers = config['num_hidden_layers']
    input_dim = train_set.n_features
    output_dim = config.get('output_dim', train_set.n_labels)
    hidden_dim = config['hidden_dim']

    model = create_model(num_hidden_layers, input_dim, output_dim, hidden_dim)

    # Set training hyperparameters
    num_epoches = config['num_epoches']
    minibatch_size = config['minibatch_size']
    learning_rate = config['learning_rate']

    # Trajectory
    acc_train_record = []
    loss_train_record = []
    acc_test_record = []
    loss_test_record = []
    best_epoch = 0
    best_model = None

    # Train and evaluate during training
    for t in range(num_epoches):
        print(f'Epoch {t}')

        # Shuffle the training data
        idx_data = np.random.permutation(n_train)

        for i in range(int(np.floor(n_train / minibatch_size))):
            x, y = train_set.sample(idx_data[i * minibatch_size: (i + 1) * minibatch_size])

            cache = forward_pass(model, x, y, calc_loss=True)
            backward_pass(model, cache)

            # Update model parameters
            mini_batch_gradient_descent(model, learning_rate)

        # Compute training accuracy and loss
        train_acc, train_loss = compute_accuracy_and_loss(train_set, model, minibatch_size)
        acc_train_record.append(train_acc)
        loss_train_record.append(train_loss)
        print(f'Training loss: {train_loss}, accuracy: {train_acc}')

        # Compute test accuracy and loss
        if evaluate:
            test_acc, test_loss = compute_accuracy_and_loss(test_set, model, minibatch_size)
            # Update best model
            if len(acc_test_record) == 0 or test_acc > max(acc_test_record):
                best_model = deepcopy(model)
                best_epoch = t
            acc_test_record.append(test_acc)
            loss_test_record.append(test_loss)
            print(f'Test loss: {test_loss}, accuracy: {test_acc}')

    if evaluate:
        timestamp = time.strftime("%m%d-%H%M")
        print(f'Test accuracy at the best epoch (epoch {best_epoch}): {acc_test_record[best_epoch]: .4f}')

        # Plot the learning curves
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(loss_train_record, label='Training Loss')
        axes[0].plot(loss_test_record, label='Test Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Cross-entropy Loss')
        axes[0].set_title('Training and Test Loss')
        axes[0].legend()

        axes[1].plot(acc_train_record, label='Training Accuracy')
        axes[1].plot(acc_test_record, label='Test Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Test Accuracy')
        axes[1].legend()

        # fig_dir = './figure/loss'
        # if not os.path.exists(fig_dir):
        #     os.makedirs(fig_dir)
        # plt.savefig(os.path.join(fig_dir, f"{timestamp}_lr_{learning_rate}_bs_{minibatch_size}.png"))
        # plt.show()

    return best_model if evaluate else model, train_max_min_price, mu_std_price


def inference(model: dict, max_min_price: tuple, mu_std_price: tuple,
              test_data_file: str = 'test_data.csv', test_label_file=None):
    test_set = DataLoader(test_data_file, label_file=None, mu_std_price=mu_std_price)
    # print(f'max_min_price: {max_min_price}')
    x = test_set.sample(np.arange(len(test_set)))
    # print(x[:10, :])
    # print(model['linear0'].params['W'].shape)
    # print(model['linear0'].params['W'][:10, :3])
    cache = forward_pass(model, x, calc_loss=False)
    # for i in range(len(cache)):
    #     print(cache[i].shape)
    #     print(cache[i][:3, :5])
    # shapes = [arr.shape for arr in cache]
    # print(shapes)
    y = predict_label(cache[-1]).reshape(-1).tolist()

    if test_label_file is not None:
        test_set = DataLoader(test_data_file, test_label_file, mu_std_price=mu_std_price)
        acc, _ = compute_accuracy_and_loss(test_set, model, minibatch_size=len(test_set))
        print(f"acc: {acc}")

    output_filename = 'output.csv'
    column_name = 'BEDS'
    with open(output_filename, 'w') as f:
        f.write(column_name + '\n')
        for item in y:
            f.write(str(item) + '\n')
