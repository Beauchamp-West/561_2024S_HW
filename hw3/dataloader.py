import csv
import numpy as np

from utils import one_hot

type_map = {
    'Co-op for sale': 0,
    'Coming Soon': 1,
    'Condo for sale': 2,
    'Contingent': 3,
    'For sale': 4,
    'Foreclosure': 5,
    'House for sale': 6,
    'Land for sale': 7,
    'Mobile house for sale': 8,
    'Multi-family home for sale': 9,
    'Pending': 10,
    'Townhouse for sale': 11,
}


class DataLoader:
    def __init__(self, data_file: str, label_file=None, max_min_price=None,
                 mu_std_price=None, max_min_sqft=None) -> None:
        # Load data from CSV files
        with open(data_file, 'r') as data_csv:
            data_csv = csv.reader(data_csv, delimiter=',')
            next(data_csv)  # Skip the first row
            data = []
            for row in data_csv:
                # Convert each element to appropriate data type
                # row = row.strip().split(',')
                # row = [float(val) if val.replace('.', '', 1).isdigit() else val for val in row]
                data.append(row)
        data = np.array(data)
        self.type: np.ndarray = data[:, 1]  # str
        self.price: np.ndarray = data[:, 2].astype(float)
        self.bath: np.ndarray = data[:, 3].astype(float)
        self.sqft: np.ndarray = data[:, 4].astype(float)

        self.have_label: bool = True if label_file is not None else False
        if self.have_label:
            with open(label_file, 'r') as label_csv:
                label_reader = csv.reader(label_csv)
                next(label_reader)  # Skip the first row
                labels = np.array([row for row in label_reader], dtype=np.uint8)
            self.label: np.ndarray = labels[:, 0]
            self.n_labels: int = int(self.label.max())
            # print(f"n_labels: {self.n_labels}")
            # print(f"min_beds: {np.min(self.label)}")
        # print(f'min bath: {np.min(self.bath)}, max bath: {np.max(self.bath)}')

        self.n_features: int = 13 + 1 + 51 + 1  # 3 + len(type_map) + 1
        self.max_price: float = np.max(self.price) if max_min_price is None else max_min_price[0]
        self.min_price: float = np.min(self.price) if max_min_price is None else max_min_price[1]
        self.mu_price: float = np.mean(self.price) if mu_std_price is None else mu_std_price[0]
        self.std_price: float = np.std(self.price) if mu_std_price is None else mu_std_price[1]
        self.max_sqft: float = np.max(self.sqft) if max_min_sqft is None else max_min_sqft[0]
        self.min_sqft: float = np.min(self.sqft) if max_min_sqft is None else max_min_sqft[1]

    def __len__(self):
        return len(self.type)

    def sample(self, indices):
        # batch_x = np.empty((len(indices), self.n_features), dtype=np.float32)
        # if self.have_label:
        #     batch_y = np.empty((len(indices), 1), dtype=int)

        types = self.type[indices]
        int_array = np.ones_like(types, dtype=int) * len(type_map)
        for key, value in type_map.items():
            int_array[types == key] = value
        types = one_hot(int_array, len(type_map))

        # prices = (self.price[indices].reshape(-1, 1) - self.min_price) / (self.max_price - self.min_price)
        prices = (self.price[indices].reshape(-1, 1) - self.mu_price) / self.std_price

        bathes = self.bath[indices].reshape(-1, 1)
        bathes = np.ceil(bathes).astype(int)
        bathes = one_hot(bathes, num_classes=50)

        sqfts = self.sqft[indices].reshape(-1, 1)
        sqfts = (sqfts - self.min_sqft) / (self.max_sqft - self.min_sqft)

        batch_x = np.concatenate((types, prices, bathes, sqfts), axis=1)
        if self.have_label:
            batch_y = self.label[indices]
            return batch_x, batch_y
        # for i in range(len(indices)):
        #     type, price, bath, sqft = type_map.get(self.type[indices[i]], 0.), self.price[indices[i]], self.bath[
        #         indices[i]], self.sqft[indices[i]]
        #     batch_x[i, :] = np.concatenate((type, price, bath, sqft), axis=None)
        #     # price, bath, sqft = self.price[indices[i]], self.bath[indices[i]], self.sqft[indices[i]]
        #     # batch_x[i, :] = np.concatenate((price, bath, sqft), axis=None)
        #     if self.have_label:
        #         batch_y[i, 0] = self.label[indices[i]]
        return batch_x
