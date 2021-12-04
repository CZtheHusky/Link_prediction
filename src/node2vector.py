import datetime

import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
import torch.utils.data as Data
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import random
from sklearn.metrics import roc_auc_score

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def tensor_hook(grad):
    print('tensor hook')
    print('grad:', grad)
    return grad


class Node2Vector:
    def __init__(self, G, G_full, edge_val, edge_train, dimensions, walks_per_node, walk_length, context_size, parma_p,
                 parma_q, val_ratio, lr, neg_ratio):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        np.random.seed(100)
        self.neg_ratio = neg_ratio
        self.G = G
        self.G_full = G_full
        self.dimensions = dimensions
        self.parma_p = parma_p
        self.parma_q = parma_q
        self.walks_per_node = walks_per_node
        self.walk_length = walk_length
        self.context_size = context_size
        self.edge_train = edge_train
        self.edge_val = edge_val
        self.alias_nodes, self.alias_edges = self.altered_transition_prob()
        self.val_ratio = val_ratio
        self.lr = lr
        self.idx2wd = [i for i in G_full.nodes]
        self.wd2idx = defaultdict(int)
        for idx, nodes in enumerate(self.idx2wd):
            self.wd2idx[nodes] = idx
        self.num_words = len(G_full.nodes)
        self.num_embeddings = int(self.num_words / 10)
        self.train()

    def random_walk(self):
        print('random walking')
        G = self.G
        r = self.walks_per_node
        node_list = list(G.nodes)
        random.seed(datetime.datetime.now())
        random.shuffle(node_list)
        walks = []
        for i in range(r):
            for j in tqdm(node_list):
                walks.append(self.n2v_walk(j))
        return walks

    def alias_table(self, ratio):
        l = len(ratio)
        accept, alias = [0] * l, [0] * l
        small, large = [], []
        for i, prob in enumerate(ratio):
            if prob < 1:
                small.append(i)
            elif prob > 1:
                large.append(i)
            else:
                accept[i] = 1
        while len(small) and len(large):
            index_s, index_l = small.pop(), large.pop()
            accept[index_s] = ratio[index_s]
            alias[index_s] = index_l
            ratio[index_l] = ratio[index_l] - 1 + ratio[index_s]
            if ratio[index_l] < 1:
                small.append(index_l)
            elif ratio[index_l] > 1:
                large.append(index_l)
            else:
                accept[index_l] = 1
        while len(large):
            index_l = large.pop()
            accept[index_l] = 1
        while len(small):
            index_s = small.pop()
            accept[index_s] = 1
        return accept, alias

    def alias_sampling(self, accept, alias):

        N = len(accept)
        r = np.random.random()
        i = int(np.random.random() * N)
        if r < accept[i]:
            return i
        else:
            return alias[i]

    def n2v_walk(self, start):
        walk_length = self.walk_length
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges
        walk = [start]
        while len(walk) < walk_length:
            current = walk[-1]
            cur_neighbours = list(G.neighbors(current))
            if len(cur_neighbours) > 0:
                if len(walk) == 1:
                    walk.append(
                        cur_neighbours[self.alias_sampling(alias_nodes[current][0], alias_nodes[current][1])])
                else:
                    prev_node = walk[-2]
                    edge = (prev_node, current)
                    next_node = cur_neighbours[self.alias_sampling(alias_edges[edge][0], alias_edges[edge][1])]
                    walk.append(next_node)
            else:
                break
        return walk

    def alias_edge_generation(self, t, v):
        G = self.G
        p = self.parma_p
        q = self.parma_q
        probs = []
        for x in G.neighbors(v):
            weight = G[v][x].get('weight', 1.0)
            if x == t:
                probs.append(weight / p)
            elif G.has_edge(x, t):
                probs.append(weight)
            else:
                probs.append(weight / q)
        total = sum(probs)
        normalized_probs = [prob / total for prob in probs]
        return self.alias_table(normalized_probs)

    def altered_transition_prob(self):
        G = self.G
        alias_nodes = {}
        for node in G.nodes():
            probs = [G[node][neighbour].get('weight', 1.0) for neighbour in G.neighbors(node)]
            total = sum(probs)
            normalized_probs = [prob / total for prob in probs]
            alias_nodes[node] = self.alias_table(normalized_probs)
        alias_edges = {}
        for edge in G.edges():
            alias_edges[(edge[0], edge[1])] = self.alias_edge_generation(edge[0], edge[1])
            alias_edges[(edge[1], edge[0])] = self.alias_edge_generation(edge[1], edge[0])
        return alias_nodes, alias_edges

    def negative_sample_from_word(self, center, neg_num, neg_table):
        counter = 0
        neg_sp = []
        idx2wd = self.idx2wd
        while counter < neg_num:
            tmp = random.randint(0, len(neg_table) - 1)
            if idx2wd[neg_table[tmp]] not in self.G_full.neighbors(idx2wd[center]) and neg_table[tmp] not in neg_sp:
                neg_sp.append(neg_table[tmp])
                counter += 1
        return neg_sp

    def negative_sample(self, neg_num, node_list, neg_table):
        counter = 0
        neg_sp = []
        random.seed(datetime.datetime.now())
        idx2wd = self.idx2wd
        while counter != neg_num:
            tmp = random.randint(0, len(node_list) - 1)
            tmp1 = random.randint(0, len(neg_table) - 1)
            if (idx2wd[neg_table[tmp1]], node_list[tmp]) not in self.G_full.edges:
                neg_sp.append([idx2wd[neg_table[tmp1]], node_list[tmp]])
                counter += 1
        return neg_sp

    def validation_prepare(self, neg_table):
        EV = self.edge_val
        w2i = self.wd2idx
        neg_ratio = 5
        input_x_EV = np.zeros(len(EV))
        input_y_EV = np.zeros(len(EV))
        neg_x = np.zeros(neg_ratio * len(EV))
        neg_y = np.zeros(neg_ratio * len(EV))
        neg_sample = self.negative_sample(neg_ratio * len(EV), list(self.G_full.nodes), neg_table)
        for i in range(len(EV)):
            input_x_EV[i] = w2i[EV[i][0]]
            input_y_EV[i] = w2i[EV[i][1]]
        for i in range(neg_ratio * len(EV)):
            neg_x[i] = w2i[neg_sample[i][0]]
            neg_y[i] = w2i[neg_sample[i][1]]
        self.input_y_EV = input_y_EV
        self.input_x_EV = input_x_EV
        self.neg_x = neg_x
        self.neg_y = neg_y

    def neg_samp_table(self):
        frequency_table = np.zeros(len(self.G_full.nodes))
        wd2idx = self.wd2idx
        for sentence in self.Walks:
            for word in sentence:
                frequency_table[wd2idx[word]] += 1
        pow_freq = frequency_table ** 0.75
        words_pow = sum(pow_freq)
        ratio = pow_freq / words_pow
        count = ratio * len(ratio) * 1e3
        neg_table = np.zeros(int(len(ratio) * 1e3))
        for i in range(1, len(count)):
            count[i] += count[i - 1]
        neg_table[0:round(count[0])] = 0
        for i in range(1, len(count)):
            neg_table[round(count[i - 1]):round(count[i])] = i
        return np.int32(neg_table)

    def score(self, x, y, neg_x, neg_y):
        pos_prob = self.model.get_prob(x, y)
        neg_prob = self.model.get_prob(neg_x, neg_y)
        y_score = np.concatenate((pos_prob, neg_prob))
        y_truth = np.concatenate((np.ones(len(pos_prob)), np.zeros((len(neg_prob)))))
        auc = roc_auc_score(y_truth, y_score)
        # count = 0
        # for i in range(len(pos_prob)):
        #     for j in range(len(neg_prob)):
        #         if pos_prob[i] > neg_prob[j]:
        #             count+=1
        # auc1 = count/len(pos_prob)/len(neg_prob)
        print('auc on val:', auc)
        return auc

    def data_generation(self):
        context_size = self.context_size
        wd2idx = self.wd2idx
        center = []
        context = []
        neg_words = []
        neg_num = self.neg_ratio
        random.seed(datetime.datetime.now())
        if not os.path.exists(
                '../data/context_' + str(int(10 * self.val_ratio)) + '_' + str(self.walks_per_node) + '_' + str(
                    self.walk_length) + '.npy') and not os.path.exists(
            '../data/neg_words_' + str(int(10 * self.val_ratio)) + '_' + str(self.walks_per_node) + '_' + str(
                self.walk_length) + '_' + str(neg_num) + '.npy'):
            self.Walks = self.random_walk()
            neg_table = self.neg_samp_table()
            print('Generating data')
            for sentence in tqdm(self.Walks):
                for i in range(context_size, len(sentence) - context_size):
                    center_w = wd2idx[sentence[i]]
                    context_i = list(range(i - context_size, i)) + list(range(i + 1, i + context_size + 1))
                    context.append([wd2idx[sentence[k]] for k in context_i])
                    center.append(center_w)
                    neg_words.append(self.negative_sample_from_word(center_w, neg_num, neg_table))
            context = np.array(context)
            center = np.array(center)
            neg_words = np.array(neg_words)
            np.save('../data/context_' + str(int(10 * self.val_ratio)) + '_' + str(self.walks_per_node) + '_' + str(
                self.walk_length) + '.npy', context)
            np.save('../data/center_' + str(int(10 * self.val_ratio)) + '_' + str(self.walks_per_node) + '_' + str(
                self.walk_length) + '.npy', center)
            np.save(
                '../data/neg_words_' + str(int(10 * self.val_ratio)) + '_' + str(self.walks_per_node) + '_' + str(
                    self.walk_length) + '_' + str(neg_num) + '.npy',
                neg_words)
            np.save(
                '../data/neg_table_' + str(int(10 * self.val_ratio)) + '_' + str(self.walks_per_node) + '_' + str(
                    self.walk_length) + '.npy',
                neg_table)
        elif not os.path.exists(
                '../data/neg_words_' + str(int(10 * self.val_ratio)) + '_' + str(self.walks_per_node) + '_' + str(
                    self.walk_length) + '_' + str(neg_num) + '.npy'):
            context = np.load('../data/context_' + str(int(10 * self.val_ratio)) + '_' + str(self.walks_per_node) + '_' + str(
                self.walk_length) + '.npy')
            center = np.load('../data/center_' + str(int(10 * self.val_ratio)) + '_' + str(self.walks_per_node) + '_' + str(
                self.walk_length) + '.npy')
            neg_table = np.load(
                '../data/neg_table_' + str(int(10 * self.val_ratio)) + '_' + str(self.walks_per_node) + '_' + str(
                    self.walk_length) + '.npy')
            print('generating neg data')
            for i in tqdm(center):
                neg_words.append(self.negative_sample_from_word(i, neg_num, neg_table))
            np.save(
                '../data/neg_words_' + str(int(10 * self.val_ratio)) + '_' + str(self.walks_per_node) + '_' + str(
                    self.walk_length) + '_' + str(neg_num) + '.npy',
                neg_words)
        else:
            print('loading data')
            context = np.load('../data/context_' + str(int(10 * self.val_ratio)) + '_' + str(self.walks_per_node) + '_' + str(
                self.walk_length) + '.npy')
            center = np.load('../data/center_' + str(int(10 * self.val_ratio)) + '_' + str(self.walks_per_node) + '_' + str(
                self.walk_length) + '.npy')
            neg_words = np.load(
                '../data/neg_words_' + str(int(10 * self.val_ratio)) + '_' + str(self.walks_per_node) + '_' + str(
                    self.walk_length) + '_' + str(neg_num) + '.npy')
            neg_table = np.load(
                '../data/neg_table_' + str(int(10 * self.val_ratio)) + '_' + str(self.walks_per_node) + '_' + str(
                    self.walk_length) + '.npy')
        return context, center, neg_words, neg_table

    def train(self):
        d = self.dimensions
        num_words = self.num_words
        batch_size = 1000
        context, center, neg_words, neg_table = self.data_generation()
        context, center, neg_words = torch.LongTensor(context), torch.LongTensor(center), torch.LongTensor(neg_words)
        self.validation_prepare(neg_table)
        dataset = Data.TensorDataset(context, center, neg_words)
        loader = Data.DataLoader(dataset, batch_size, True, num_workers=6)
        self.model = word2vector(num_words, d).to(self.device)
        # self.model.register_backward_hook(self.model.my_hook)
        optimizer = optim.SparseAdam(params=self.model.parameters(), lr=self.lr)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(loader))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, verbose=1, min_lr=1e-8,
                                                               patience=1000)
        print('training')
        counter = 0
        loss_val = []
        x = []
        epo = []
        auc_value = []
        for epoch in range(1):
            for i, (context, center, neg_words) in enumerate(tqdm(loader)):
                context = context.to(self.device)
                center = center.to(self.device)
                neg_words = neg_words.to(self.device)
                # batch_x.requires_grad = True
                # batch_x.register_hook(tensor_hook)
                loss = self.model(center, context, neg_words)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step(loss)
                # print('input.grad:', batch_x.grad)
                # for param in self.model.parameters():
                #     print('{}:grad->{}'.format(param, param.grad))
                loss_val.append(loss.item())
                x.append(counter)
                counter += 1
            self.model.prob_init()
            auc = self.score(self.input_x_EV, self.input_y_EV, self.neg_x, self.neg_y)
            auc_value.append(auc)
            epo.append(epoch)
        plt.figure(dpi=300, figsize=(24, 8))
        plt.plot(x, loss_val)
        plt.savefig('../data/loss_valratio_' + str(self.val_ratio) + '.png', bbox_inches='tight')
        plt.clf()
        plt.figure(dpi=300, figsize=(24, 8))
        plt.plot(epo, auc_value)
        plt.savefig('../data/auc_valratio_' + str(self.val_ratio) + '.png', bbox_inches='tight')
        plt.clf()


class word2vector(nn.Module):
    def __init__(self, num_words, d):
        super(word2vector, self).__init__()
        self.W = nn.Embedding(num_words, d, sparse=True)
        self.V = nn.Embedding(num_words, d, sparse=True)
        init_range = 1.0 / d
        nn.init.uniform_(self.W.weight.data, -init_range, init_range)
        nn.init.constant_(self.V.weight.data, 0)
        self.logsigmoid = nn.LogSigmoid()

    def forward(self, pos_input, pos_target, neg):
        in_ = self.W(pos_input)
        target = self.V(pos_target)
        neg_in = self.V(neg)
        score = torch.bmm(target, in_.unsqueeze(2)).squeeze()
        score = torch.clamp(score, max=10, min=-10)
        score = -torch.sum(self.logsigmoid(score), dim=1)
        neg_score = torch.bmm(neg_in, in_.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(self.logsigmoid(-neg_score), dim=1)
        return torch.mean(score + neg_score)

    def prob_init(self):
        embedding = self.W.weight
        self.score_map1 = torch.matmul(embedding, embedding.T)
        self.score_map2 = torch.sum(embedding * embedding, dim=1)

    def get_prob(self, x, y):
        h0 = self.score_map1[x, y]
        h1 = torch.sqrt(self.score_map2[x])
        h2 = torch.sqrt(self.score_map2[y])
        res = h0 / h1 / h2
        res = res.to('cpu').detach().numpy()
        res = (res + 1) / 2
        return res

    # class word2vector(nn.Module):
    #     def __init__(self, num_words, d):
    #         super(word2vector, self).__init__()
    #         self.W = nn.Parameter(torch.rand(num_words, d))
    #         self.V = nn.Parameter(torch.rand(d, num_words))
    #         self.softmax = nn.Softmax(dim=1)
    #
    #     def forward(self, x):
    #         x = torch.matmul(x, self.W)
    #         x = torch.matmul(x, self.V)
    #         x = self.softmax(x)
    #         return x
    #
    #     def word_vector(self):
    #         return self.W
    #
    #     def my_hook(self, module, grad_input, grad_output):
    #         print('doing my_hook')
    #         print('original grad:', grad_input)
    #         print('original outgrad:', grad_output)
    #         # grad_input = grad_input[0]*self.input   # 这里把hook函数内对grad_input的操作进行了注释，
    #         # grad_input = tuple([grad_input])        # 返回的grad_input必须是tuple，所以我们进行了tuple包装。
    #         # print('now grad:', grad_input)
    #         return grad_input

    # class word2vector(nn.Module):
    #     def __init__(self, num_embeddings, num_words, final_dim):
    #         super(word2vector, self).__init__()
    #         self.W1 = nn.Parameter(torch.randn(num_embeddings, num_words))
    #         self.V1 = nn.Parameter(torch.randn(num_words, final_dim))
    #         self.W2 = nn.Parameter(torch.randn(num_embeddings, num_words))
    #         self.V2 = nn.Parameter(torch.randn(num_words, final_dim))
    #         self.relu = nn.LeakyReLU()
    #
    #     def forward(self, xy):
    #         # h1 = self.W1.data[xy[:, 0]]
    #         # h2 = self.W2.data[xy[:, 1]]
    #         h1 = self.relu(self.W1.data[xy[:, 0]])
    #         h2 = self.relu(self.W2.data[xy[:, 1]])
    #         out1 = torch.matmul(h1, self.V1)
    #         out2 = torch.matmul(h2, self.V2)
    #         return out1, out2
    #
    #     def inteference(self, x, y):
    #         x = x.long()
    #         y = y.long()
    #         h1 = self.W1.data[x]
    #         h2 = self.W2.data[y]
    #         h1 = self.relu(h1)
    #         h2 = self.relu(h2)
    #         out1 = torch.matmul(h1, self.V1)
    #         out2 = torch.matmul(h2, self.V2)
    #         out = torch.cosine_similarity(out1, out2, dim=1)
    #         return out