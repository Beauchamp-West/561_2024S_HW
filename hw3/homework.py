import time

from neural_networks import train_and_eval, inference

# USE_DATASET_SPLIT 1

config = {
    'train_data_file': 'train_data.csv',  # "resource/train_data1.csv",
    'test_data_file': "resource/test_data1.csv",
    'train_label_file': 'train_label.csv',  # "resource/train_label1.csv",
    'test_label_file': "resource/test_label1.csv",
    'num_hidden_layers': 8,
    'hidden_dim': 128,
    'output_dim': 50,
    'num_epoches': 100,
    'minibatch_size': 16,
    'learning_rate': 1e-2,
    'random_seed': 37,
}
test_data_file = 'test_data.csv'
test_label_file = None
# config = {
#     'train_data_file': "resource/train_data1.csv",
#     'test_data_file': "resource/test_data1.csv",
#     'train_label_file': "resource/train_label1.csv",
#     'test_label_file': "resource/test_label1.csv",
#     'num_hidden_layers': 8,
#     'hidden_dim': 128,
#     'output_dim': 50,
#     'num_epoches': 100,
#     'minibatch_size': 16,
#     'learning_rate': 1e-2,
#     'random_seed': 37,
# }
# test_data_file = "resource/test_data1.csv"
# test_label_file = "resource/test_label1.csv"


if __name__ == "__main__":
    start = time.time()
    # for i in range(1, 6):
    #     data_file = f'resource/train_data{i}.csv'
    #     data = DataLoader(data_file)
    #     data_file = f'resource/test_data{i}.csv'
    #     data = DataLoader(data_file)
    model, train_max_min_price, train_mu_std_price = train_and_eval(config=config, evaluate=False)
    inference(model, train_max_min_price, train_mu_std_price,
              test_data_file=test_data_file, test_label_file=test_label_file)

    print(f"Time taken: {time.time() - start: .4f} s")
