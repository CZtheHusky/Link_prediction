import random
import networkx as nx
import csv

import numpy as np

from node2vector import *
import pickle
import pandas as pd


class link_prediction():
    def __init__(self, filename, val_ratio, dimensions, walks_per_node, walk_length, context_size, parama_p, parama_q,
                 lr, test_path, neg_ratio):
        self.graph_train, self.graph_full, self.edge_train, self.edge_val = self.graph_build(filename, val_ratio)
        self.n2v = Node2Vector(self.graph_train, self.graph_full, self.edge_val, self.edge_train, dimensions,
                               walks_per_node, walk_length, context_size, parama_p, parama_q, val_ratio, lr, neg_ratio)
        self.predict(test_path)

    def graph_build(self, filename, val_ratio):
        GT = nx.Graph()
        G_full = nx.Graph()
        edges = []
        with open(filename, encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                edges.append([row[0], row[1]])
                G_full.add_edge(row[0], row[1])
        EV = edges[:int(val_ratio * len(edges))]
        ET = edges[int(val_ratio * len(edges)):]
        for i in ET:
            GT.add_edge(i[0], i[1])
        return GT, G_full, ET, EV

    def predict(self, test_path):
        n2v = self.n2v
        wd2idx = n2v.wd2idx
        result = defaultdict(int)
        with open(test_path, encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            input_x = []
            input_y = []
            for row in reader:
                input_x.append(wd2idx[row[1]])
                input_y.append(wd2idx[row[2]])
            res = n2v.model.get_prob(np.array(input_x), np.array(input_y))
            for i in range(len(input_y)):
                result[i] = res[i]
        res = pd.DataFrame(list(result.items()), columns=None)
        res.to_csv('../Prediction.csv', index=False, header=None)


if __name__ == '__main__':
    filename = '../data/lab2_edge.csv'
    test_data = '../data/lab2_test.csv'
    dimensions = 300
    walks_per_node = 20
    walk_length = 25
    context_size = 1
    parama_p = 4
    parama_q = 1
    val_ratio = 0.1
    lr = 1e-4
    neg_ratio = 5
    LP = link_prediction(filename, val_ratio, dimensions, walks_per_node, walk_length, context_size, parama_p, parama_q,
                         lr, test_data, neg_ratio)
