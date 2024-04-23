import json
import os
import time
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

from hw3.dataloader import DataLoader
from hw3.utils import mini_batch_gradient_descent
from neural_networks import forward_pass, backward_pass, compute_accuracy_and_loss, create_model

config = {
    'num_hidden_layers': 2,
    'hidden_dim': 128,
    'output_dim': 50,
    'num_epoches': 100,
    'minibatch_size': 64,
    'learning_rate': 1e-3,
    'random_seed': 37,
}


def hyperparameter_explore(dataset_index: int, hp_name: str, hp_values: Optional[list] = None):
    # Set random seed
    seed = config.get('random_seed', None)
    np.random.seed(seed)

    # Prepare dataset
    train_data_file = f'resource/train_data{dataset_index}.csv'
    test_data_file = f'resource/test_data{dataset_index}.csv'
    train_label_file = f'resource/train_label{dataset_index}.csv'
    test_label_file = f'resource/test_label{dataset_index}.csv'

    train_set = DataLoader(train_data_file, train_label_file)
    mu_std_price = (train_set.mu_price, train_set.std_price)
    test_set = DataLoader(test_data_file, test_label_file, mu_std_price=mu_std_price)
    n_train = len(train_set)

    records = {}
    best_acc = []
    for val in hp_values:
        # Build MLP
        num_hidden_layers = config['num_hidden_layers']
        input_dim = train_set.n_features
        output_dim = config.get('output_dim', train_set.n_labels)
        hidden_dim = val  # config['hidden_dim']

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

        # Train and evaluate during training
        for t in range(num_epoches):
            # print(f'Epoch {t}')

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
            # print(f'Training loss: {train_loss}, accuracy: {train_acc}')

            # Compute test accuracy and loss
            test_acc, test_loss = compute_accuracy_and_loss(test_set, model, minibatch_size)
            # Update best model
            if len(acc_test_record) == 0 or test_acc > max(acc_test_record):
                best_epoch = t
            acc_test_record.append(test_acc)
            loss_test_record.append(test_loss)
            # print(f'Validation loss: {test_loss}, accuracy: {test_acc}')

        print(f'{hp_name}: {val}')
        acc = acc_test_record[best_epoch]
        best_acc.append(acc)
        print(f'Validation accuracy at the best epoch (epoch {best_epoch}): {acc: .4f}')

        record = {
            'train_acc': acc_train_record,
            'train_loss': loss_train_record,
            'test_acc': acc_test_record,
            'test_loss': loss_test_record,
        }
        records[val] = record

    # save records
    record_dir = ('./records')
    os.makedirs(record_dir, exist_ok=True)
    json.dump(records, open(os.path.join(record_dir, f'{hp_name}_{dataset_index}.json'), 'w'))

    return best_acc


def plot_curves(hp_name: str, hp_list: list):
    # Plot the learning curves
    fig, axes = plt.subplots(5, len(hp_list), figsize=(4 * len(hp_list), 15))
    for idx in range(5):
        records_filename = f'./records/{hp_name}_{idx + 1}.json'
        records = json.load(open(records_filename))
        for i, hp_value in enumerate(hp_list):
            record = records[str(hp_value)]
            # axes[idx, i].plot(record['train_loss'], label='Training Loss')
            # axes[idx, i].plot(record['test_loss'], label='Validation Loss')
            axes[idx, i].plot(record['train_acc'], label='Training Accuracy')
            axes[idx, i].plot(record['test_acc'], label='Validation Accuracy')

            # axes[idx, i].set_xlabel('Epoch')
            # axes[idx, i].set_ylabel('Cross-entropy Loss')
            # axes[idx, i].set_title(f'Dataset {idx+1} {hp_name.upper()} {hp_value} Loss')
            # axes[idx, i].legend()
            axes[idx, i].set_xlabel('Epoch')
            axes[idx, i].set_ylabel('Accuracy')
            axes[idx, i].set_title(f'Dataset {idx + 1} {hp_name.upper()} {hp_value}')
            axes[idx, i].legend()

    fig_dir = './figure'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    fig.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{hp_name}.png"))
    plt.show()


BATCH_SIZE = [16, 32, 64, 128]
LR = [1e-2, 1e-3, 1e-4]
NUM_HIDDEN_LAYERS = [2, 4, 8]
HIDDEN_DIM = [64, 128, 256]

if __name__ == "__main__":
    start = time.time()

    acc = []
    # Batch size
    for i in range(1, 6):
        best_acc = hyperparameter_explore(dataset_index=i, hp_name='hidden_dim', hp_values=HIDDEN_DIM)
        acc.append(best_acc)
    acc_mean = np.mean(np.array(acc), axis=0)
    print(f'average accuracy: {acc_mean}')

    plot_curves('hidden_dim', HIDDEN_DIM)
    print(f"Time taken: {time.time() - start: .4f} s")
