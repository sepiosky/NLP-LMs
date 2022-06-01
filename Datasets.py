import numpy as np
import torch.utils.data as D
import itertools
import torch

class MasnaviDataset(D.Dataset):
    def __init__(self, masnavi_beyts, corpus):
        self.masnavi_beyts = masnavi_beyts
        self.corpus_index = {}
        for idx, w in enumerate(corpus):
            self.corpus_index[w]=idx 
        self.pad_len = max([len(m) for m in list(itertools.chain(*masnavi_beyts))])+2 # +2 is for bom and eom
    def __len__(self):
            return len(self.masnavi_beyts)
    def __getitem__(self, idx):
            mesra = ["__BOM__"]+self.masnavi_beyts[idx][0]+["__EOM__"]
            mesra_padded = ["__PAD__"]*(self.pad_len-len(mesra)) + mesra
            
            target = ["__BOM__"]+self.masnavi_beyts[idx][1]+["__EOM__"]
            target_padded = target + ["__PAD__"]*(self.pad_len-len(target))
            target_indices = torch.LongTensor([self.corpus_index[w] for w in target_padded])
            
            #mask = torch.Tensor([1]*len(target) + [0]*(self.pad_len-len(target)))
            mask = len(target)
            return ("#".join(mesra_padded), ("#".join(target_padded), target_indices, mask))

class RhymeBatchSampler(object):
    def __init__(self, rhymes, npratio, iterations, batch_size):
        super(RhymeBatchSampler, self).__init__()
        self.npratio = npratio # npratio=1/3 means 1/3 of dataset are positive rhymes
        self.iterations = iterations
        self.batch_size = batch_size
        mask = [int(x[2]==1) for x in rhymes]
        self.pindices = np.nonzero(mask)[0]
        self.nindices = np.nonzero([1-r for r in mask])[0]
        self.positive_rhymes_size = int(self.npratio*self.batch_size)
    def __iter__(self):
        for _ in range(self.iterations):
            prhymes = np.random.choice(self.pindices, self.positive_rhymes_size)
            nrhymes = np.random.choice(self.nindices, self.batch_size-self.positive_rhymes_size)

            total_batch_indexes = np.append(prhymes, nrhymes)
            np.random.shuffle(total_batch_indexes)

            yield total_batch_indexes.astype(int)

    def __len__(self):
        return self.iterations