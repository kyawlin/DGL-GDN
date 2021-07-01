import pandas as pd
import torch
import dgl
from dgl.data import DGLDataset


class TimeDataset(DGLDataset):
    def __init__(self, is_train, config):

        self.is_train = is_train
        self.config = config
        super().__init__(name="datase-msl")

    def process(self):

        if self.is_train:
            self.df = pd.read_csv("./data/msl/train.csv", sep=",", index_col=0)
        else:
            self.df = pd.read_csv("./data/msl/test.csv", sep=",", index_col=0)

        self.features_name = self.df.columns.tolist()
        self.features_set = set([x.strip() for x in self.features_name])
        self.structure_map = dict.fromkeys(self.features_set, [])

        u, v = [], []

        for i in self.features_set:
            self.structure_map[i] = self.features_set - {i}

        for node_name, node_list in self.structure_map.items():
            p_idx = self.features_name.index(node_name)
            for child in node_list:
                child_idx = self.features_name.index(child)
                u.append(child_idx)
                v.append(p_idx)

        self.graph = dgl.graph((u, v))

        slide_win, slide_stride = [
            self.config[k] for k in ["slide_win", "slide_stride"]
        ]
        data = []

        for feature in self.features_set:
            if feature in self.df.columns:
                data.append(self.df.loc[:, feature].values.tolist())
            else:
                print(feature, "not exist in data")

        sample_n = len(data[0])

        if self.is_train:
            data.append([0] * sample_n)  # no attack
        else:
            data.append(self.df.attack.tolist())

        labels = torch.tensor(data[-1]).double()
        if self.is_train:
            data = torch.tensor(data[:-1]).double()
        else:
            data = torch.tensor(data[:-2]).double()

        total_time_len = data.shape[1]

        if self.is_train:
            sliding_range = range(slide_win, total_time_len, slide_stride)
        else:
            sliding_range = range(slide_win, total_time_len)

        x_arr, y_arr = [], []
        labels_arr = []

        for i in sliding_range:
            ft = data[:, i - slide_win : i]
            tar = data[:, i]
            x_arr.append(ft)
            y_arr.append(tar)
            labels_arr.append(labels[i])

        self.x = torch.stack(x_arr).contiguous()
        self.y = torch.stack(y_arr).contiguous()
        self.labels = torch.Tensor(labels_arr).contiguous()

    def __getitem__(self, i):

        feature = self.x[i].double()
        y = self.y[i].double()
        label = self.labels[i].double()

        return feature, y, label

    def getGraph(self):
        return self.graph

    def __len__(self):

        return len(self.x)  # ,self.graph.num_nodes(),self.graph.num_edges()
