import torch
import torch.nn as nn
import numpy as np
import pickle


def ham_dist(x, y):
    """
        x   :   [n1 x C]
        y   :   [n2 x C]
        ret :   [n1 x n2]
    """
    return torch.cdist(x, y, p=0.)


def weighted_ham_dist(x, y, weight):
    """
        x   :   [n1 x C]
        y   :   [n2 x C]
        weight: [C]

        ret :   [n1 x n2]
    """
    if len(x.shape) == 1:
        x = x.unsqueeze(0)
    if len(y.shape) == 1:
        y = y.unsqueeze(0)
    C = len(weight)
    x = x.unsqueeze(1).repeat(1, len(y), 1)
    y = y.unsqueeze(0)
    weighted = torch.logical_xor(x, y) * weight.reshape(1, 1, C)
    return torch.mean(weighted, dim=-1)


def ham_score(label, weight):
    """
        label   :  [n x C]
        weight  :  [C]

        return  :   n
    """
    return torch.sum(label * weight.unsqueeze(0), dim=-1)


class MemoryBanks(nn.Module):
    """
        Memory Bank for all objects
        Actually a shell for each class
    """
    def __init__(self, max_size, feature_dim, age_weight, freq_path="./vcoco_verb_objwise_freq.pkl"):
        super().__init__()
        self.age_weight = age_weight
        self.obj_list, self.label_weights, self.sizes_dict = self._get_weights(freq_path)
        self.banks = nn.ModuleDict({str(o):MemoryBank(max_size, feature_dim, self.label_weights[o], age_weight, o) for o in self.obj_list})

    def _get_weights(self, freq_path):
        with open(freq_path, "rb") as f:
            freq_dict = pickle.load(f)

        weights_dict = {
            k: 1. / np.array(list(v.values())) for k, v in freq_dict.items()
        }
        sizes_dict = {}
        for k, v in freq_dict.items():
            v = np.sum(list(v.values()))
            sizes_dict[k] = 16
        return list(weights_dict.keys()), weights_dict, sizes_dict


    def read(self, feature, label, obj_label, k):
        """
            feature     : n x d
            label       : [[...], [......], [.....]]
            obj_label   : [n]
        """
        assert len(feature) == len(label) == len(obj_label)

        ret_feat_list = []
        ret_label_list = []
        ret_obj_list = []
        counts = []
        for i, o in enumerate(obj_label):
            if str(o.item()) in self.banks.keys(): # why non-exist error??
                ret_feat, ret_label = self.banks[str(o.item())].read(feature[i].unsqueeze(0), label[i], k)
            
                if ret_feat is not None:
                    ret_feat_list.append(ret_feat)
                    ret_label_list.extend(ret_label)
                    ret_obj_list.append(o.item())
                    counts.append(ret_feat.shape[0])

        if len(ret_feat_list):
            return torch.cat(ret_feat_list, dim=0), ret_label_list, ret_obj_list, counts
        else:
            return None, None, None, None


    def write(self, feature, label, obj_label, n_epoch, n_iter):
        """
            feature     : n x d
            label       : [[...], [......], [.....]]
            obj_label   : [n]
        """
        assert len(feature) == len(label) == len(obj_label)
        written = {o:False for o in self.obj_list}
        for i, o in enumerate(obj_label):
            if str(o.item()) in self.banks.keys() and o.item() in written:
                if not written[o.item()]: 
                    written[o.item()] = True
                    self.banks[str(o.item())].write(feature[i].unsqueeze(0), label[i], n_epoch, n_iter)

        pointer_sizes = [v.pointer for k, v in self.banks.items()]


class MemoryBank(nn.Module):
    """
        Memory Bank for each Object
    """

    def __init__(self, max_size, feature_dim, label_weights, age_weight, o_index):
        """
            max_size:       int, the maximum size of the feature bank
            feature_dim:    int, the dimension of feature
            label_weight:   [float], the weight of each verb associated with this object
        """
        
        super().__init__()
        self.C = len(label_weights)
        self.age_weight = age_weight
        self.o_index = o_index

        self.label_weights_np = label_weights
        idx = 2 if len(label_weights) > 5 else 1
        if idx >= len(label_weights): # ugly if else to be compatible with vcoco
            self.threshold = 0
        else:
            self.threshold = sorted(self.label_weights_np)[idx]
        self.feat_buffer = nn.Parameter(torch.zeros(max_size, feature_dim), requires_grad=False)
        self.label_buffer = nn.Parameter(torch.zeros(max_size, self.C), requires_grad=False)
        self.ham_score_buffer = nn.Parameter(torch.zeros(max_size, dtype=torch.double), requires_grad=False)# should be double othewise error
        self.label_weights = nn.Parameter(torch.from_numpy(label_weights), requires_grad=False)
        self.age_buffer = nn.Parameter(torch.zeros(max_size, 2), requires_grad=False)

        self.max_size = max_size
        self.pointer = 0
        self.k = 1  


    def read(self, feature, label, K):
        if self.pointer < K:
            return None, None

        ret_feat_list = [feature]
        ret_label_list = [label.unsqueeze(0)]
        n_select = 0
        while n_select < K: 
            cur_labels = torch.cat(ret_label_list, dim=0)
            dists = weighted_ham_dist(cur_labels, self.label_buffer[:self.pointer, :], self.label_weights)
            ave_dist = torch.mean(dists, dim=0)
            sel_index = torch.topk(ave_dist, self.k)[1]
            ret_feat_list.append(self.feat_buffer[sel_index, :].clone())
            ret_label_list.append(self.label_buffer[sel_index, :].clone())
            n_select += len(sel_index)

        ret_feat = torch.cat(ret_feat_list, 0)
        ret_label = torch.cat(ret_label_list, 0) 
        return ret_feat, ret_label
            
    def write(self, feature, label, n_epoch, n_iter):
        n_data = feature.shape[0]

        # the buffer is not full
        if self.pointer < self.max_size:
            n_can_store = self.max_size - self.pointer
            if self.pointer + n_data < self.max_size:
                # still many space, just append!
                self.feat_buffer[self.pointer:self.pointer+n_data, :] = feature.detach().clone()
                self.label_buffer[self.pointer:self.pointer+n_data, :] = label.clone()
                self.ham_score_buffer[self.pointer:self.pointer+n_data] = ham_score(label, self.label_weights).clone()
                self.age_buffer[self.pointer:self.pointer+n_data, 0] = n_epoch
                self.age_buffer[self.pointer:self.pointer+n_data, 1] = n_iter
                self.pointer += n_data
            else:
                self.feat_buffer[self.pointer:self.pointer+n_can_store, :] = feature.detach().clone()
                self.label_buffer[self.pointer:self.pointer+n_can_store, :] = label.clone()
                self.ham_score_buffer[self.pointer:self.pointer+n_can_store] = ham_score(label, self.label_weights).clone()
                self.age_buffer[self.pointer:self.pointer+n_can_store, 0] = n_epoch
                self.age_buffer[self.pointer:self.pointer+n_can_store, 1] = n_iter

                self.pointer = self.max_size

        else:
            ham_score_ = ham_score(label, self.label_weights)
            if ham_score_ < self.threshold:
                return

            age_score = self.age_buffer[:, 0] * 10000 + self.age_buffer[:, 1]
            _, old_indices  = torch.topk(-age_score, self.pointer)
            
            scores = torch.zeros(self.pointer).to(self.age_buffer.device)
            scores[old_indices] += torch.arange(self.pointer).to(self.age_buffer.device)
            _, indices = torch.topk(-scores, n_data)

            self.feat_buffer[indices, :] = feature.detach().clone()
            self.label_buffer[indices, :] = label.clone()
            self.ham_score_buffer[indices] = ham_score(label, self.label_weights).clone()
            self.age_buffer[indices, 0] = n_epoch
            self.age_buffer[indices, 1] = n_iter