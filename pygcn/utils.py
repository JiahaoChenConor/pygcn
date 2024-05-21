import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    # 1. load data from data/cora.content, check the content (node & embedding)
    # 2. shape is (2708, 1435) [['31336', '0', '0', ..., '0', 'Neural_Networks'],] means [id, ...features..., label]
    #                                                                                      1   [1:-1]           -1
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    # 3. According to 2. extract features
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # 4. transfer label into one-hot encoding
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    # 5. id list
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # 6. id dict {id: index}
    idx_map = {j: i for i, j in enumerate(idx)}
    # 7. import (edge) info
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    # 8. convert original [id: id] => [index : index]
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    # 9. build adj matrix (the prev index mapping reducing the memory usage) shape is (2708, 2708)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # 10. build symmetric adjacency matrix (Transpose matrix，directed graph -> undirected graph)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # 11. norm features (not necessary)
    features = normalize(features)
    # 12. norm (A + I) is  (D^-1)(A + I)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    # 13. divide train, val, test
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix
        Here we use D^-1 A
        rather than D^-1/2 A D^-1/2
    """
    rowsum = np.array(mx.sum(1))    # row sum
    r_inv = np.power(rowsum, -1).flatten()  # row sum ^-1 e.g. 2 -> 1/2
    r_inv[np.isinf(r_inv)] = 0.     # avoid inf
    r_mat_inv = sp.diags(r_inv)     # put sum at diag
    mx = r_mat_inv.dot(mx)          # multiply to original => so mx.sum(axis=1) results in 1 in each line
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
