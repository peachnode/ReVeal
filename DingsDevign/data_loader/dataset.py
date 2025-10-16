import copy
import json
import sys
import ijson.backends.yajl2 as ijson
import os
import pickle

import torch
import dgl
import numpy as np
from tqdm import tqdm

from DingsDevign.data_loader.batch_graph import GGNNBatchGraph
from DingsDevign.utils import load_default_identifiers, debug, initialize_batch


class DataEntry:
    def __init__(self, id, num_nodes, features, edges, target):
        self.id = id
        self.num_nodes = num_nodes
        self.target = target
        self.graph = dgl.graph(([], [])).to("cpu")
        self.graph = dgl.add_nodes(self.graph, self.num_nodes, data={'features': torch.FloatTensor(features)})
        edges = np.array(edges)
        self.graph = dgl.add_edges(self.graph, edges[:,0], edges[:,2], data={'etype': torch.LongTensor(edges[:,1])})
        # for s, _type, t in edges:
        #     self.graph = dgl.add_edges(self.graph, s, t, data={'etype': torch.LongTensor([int(_type)])})
        self.graph = self.graph.formats("csr")


class DataSet:
    def __init__(self, train_src, valid_src=None, test_src=None, batch_size=32, n_ident=None, g_ident=None, l_ident=None):
        self.train_examples = []
        self.valid_examples = []
        self.test_examples = []
        self.train_batches = []
        self.valid_batches = []
        self.test_batches = []
        self.batch_size = batch_size
        # edge_type_loc = '/'.join(str(train_src).split('/')[:-1]) + 'edge_types.json'
        # with open(edge_type_loc, 'r') as f:
        #     self.edge_types = json.load(f)
        # self.max_etype = len(self.edge_types)
        # self.edge_types = {}
        # TODO: Might not be the correct max etype
        self.max_etype = 17
        self.feature_size = 0
        self.n_ident, self.g_ident, self.l_ident= load_default_identifiers(n_ident, g_ident, l_ident)
        self.read_dataset_ijson(test_src, train_src, valid_src)
        self.initialize_dataset()

    def initialize_dataset(self):
        self.initialize_train_batch()
        self.initialize_valid_batch()
        self.initialize_test_batch()

    def read_dataset_from_pkl(self, test_src, train_src, valid_src):
        debug('Reading Train File!')
        for file_name in os.listdir(train_src):
            if not file_name.endswith('pkl'):
                continue
            with open(train_src + file_name, 'rb') as f:
                example = pickle.load(f)
            example.graph = example.graph.to("cuda")
            if self.feature_size == 0:
                self.feature_size = example.features.size(1)
                debug('Feature Size %d' % self.feature_size)
            self.train_examples.append(example)

        debug('Reading Valid File!')
        for file_name in os.listdir(valid_src):
            if not file_name.endswith('pkl'):
                continue
            with open(valid_src + file_name, 'rb') as f:
                example = pickle.load(f)
            example.graph = example.graph.to("cuda")
            if self.feature_size == 0:
                self.feature_size = example.features.size(1)
                debug('Feature Size %d' % self.feature_size)
            self.valid_examples.append(example)

        debug('Reading Test File!')
        for file_name in os.listdir(test_src):
            if not file_name.endswith('pkl'):
                continue
            with open(test_src + file_name, 'rb') as f:
                example = pickle.load(f)
            example.graph = example.graph.to("cuda")
            if self.feature_size == 0:
                self.feature_size = example.features.size(1)
                debug('Feature Size %d' % self.feature_size)
            self.test_examples.append(example)

    def read_dataset(self, test_src, train_src, valid_src):
        debug('Reading Train File!')
        with open(train_src) as fp:
            train_data = json.load(fp)
            for entry in tqdm(train_data):
                id = entry["file_name"].split("_")[0]
                example = DataEntry(id=id, dataset=self, num_nodes=len(entry[self.n_ident]), features=entry[self.n_ident],
                                    edges=entry[self.g_ident], target=entry[self.l_ident][0][0])
                if self.feature_size == 0:
                    self.feature_size = example.features.size(1)
                    debug('Feature Size %d' % self.feature_size)
                self.train_examples.append(example)
            del train_data
        if valid_src is not None:
            debug('Reading Validation File!')
            with open(valid_src) as fp:
                valid_data = json.load(fp)
                for entry in tqdm(valid_data):
                    id = entry["file_name"].split("_")[0]
                    example = DataEntry(id=id, dataset=self, num_nodes=len(entry[self.n_ident]),
                                        features=entry[self.n_ident],
                                        edges=entry[self.g_ident], target=entry[self.l_ident][0][0])
                    self.valid_examples.append(example)
                del valid_data
        if test_src is not None:
            debug('Reading Test File!')
            with open(test_src) as fp:
                test_data = json.load(fp)
                for entry in tqdm(test_data):
                    id = entry["file_name"].split("_")[0]
                    example = DataEntry(id=id, dataset=self, num_nodes=len(entry[self.n_ident]),
                                        features=entry[self.n_ident],
                                        edges=entry[self.g_ident], target=entry[self.l_ident][0][0])
                    if self.feature_size == 0:
                        self.feature_size = example.features.size(1)
                        debug('Feature Size %d' % self.feature_size)
                    self.test_examples.append(example)
                del test_data
    
    def read_dataset_ijson(self, test_src, train_src, valid_src):
        debug('Reading Train File!')
        with open(train_src) as fp:
            for entry in tqdm(ijson.items(fp, "item")):
                id = entry["file_name"].split("_")[0]
                example = DataEntry(id=id, num_nodes=len(entry[self.n_ident]), features=entry[self.n_ident],
                                    edges=entry[self.g_ident], target=entry[self.l_ident][0][0])
                if self.feature_size == 0:
                    self.feature_size = example.graph.ndata['features'].size(1)
                    print('Feature Size %d' % self.feature_size)
                self.train_examples.append(example)
        if valid_src is not None:
            debug('Reading Validation File!')
            with open(valid_src) as fp:
                for entry in tqdm(ijson.items(fp, "item")):
                    id = entry["file_name"].split("_")[0]
                    example = DataEntry(id=id, num_nodes=len(entry[self.n_ident]),
                                        features=entry[self.n_ident],
                                        edges=entry[self.g_ident], target=entry[self.l_ident][0][0])
                    self.valid_examples.append(example)
        if test_src is not None:
            debug('Reading Test File!')
            with open(test_src) as fp:
                for entry in tqdm(ijson.items(fp, "item")):
                    id = entry["file_name"].split("_")[0]
                    example = DataEntry(id=id, num_nodes=len(entry[self.n_ident]),
                                        features=entry[self.n_ident],
                                        edges=entry[self.g_ident], target=entry[self.l_ident][0][0])
                    self.test_examples.append(example)

    # def get_edge_type_number(self, _type):
    #     if _type not in self.edge_types:
    #         self.edge_types[_type] = self.max_etype
    #         self.max_etype += 1
    #     return self.edge_types[_type]

    @property
    def max_edge_type(self):
        return self.max_etype

    def initialize_train_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.train_batches = initialize_batch(self.train_examples, batch_size, shuffle=True)
        return len(self.train_batches)
        pass

    def initialize_valid_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.valid_batches = initialize_batch(self.valid_examples, batch_size)
        return len(self.valid_batches)
        pass

    def initialize_test_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.test_batches = initialize_batch(self.test_examples, batch_size)
        return len(self.test_batches)
        pass

    def get_dataset_by_ids_for_GGNN(self, entries, ids):
        taken_entries = [entries[i] for i in ids]
        # labels = []
        # graphs = []
        # for e in taken_entries:
        #     e.graph.ndata['features'] = e.graph.ndata['features'].to("cuda")
        #     e.graph.edata['etype'] = e.graph.edata['etype'].to("cuda")
        #     labels.append(e.target)
        #     graphs.append(e.graph)
        # batch_graph = dgl.batch(graphs)
        labels = [e.target for e in taken_entries]
        batch_graph = GGNNBatchGraph()
        for entry in taken_entries:
            batch_graph.add_subgraph(copy.deepcopy(entry.graph))
        return batch_graph, torch.FloatTensor(labels)

    def get_next_train_batch(self):
        if len(self.train_batches) == 0:
            self.initialize_train_batch()
        ids = self.train_batches.pop()
        return self.get_dataset_by_ids_for_GGNN(self.train_examples, ids)

    def get_next_valid_batch(self):
        if len(self.valid_batches) == 0:
            self.initialize_valid_batch()
        ids = self.valid_batches.pop()
        return self.get_dataset_by_ids_for_GGNN(self.valid_examples, ids)

    def get_next_test_batch(self):
        if len(self.test_batches) == 0:
            self.initialize_test_batch()
        ids = self.test_batches.pop()
        return self.get_dataset_by_ids_for_GGNN(self.test_examples, ids)
