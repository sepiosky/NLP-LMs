import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

PERSIAN_EMBEDDINGS = "اأآبپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیئ"

def myword2vec(word): #simple word to vector #TODO change format to one-hot
    vec = []
    for c in word:
        if c in PERSIAN_EMBEDDINGS:
            vec.append(PERSIAN_EMBEDDINGS.index(c))
        else:
            vec.append(len(PERSIAN_EMBEDDINGS)) # in else to result khoobe nabood
    vec = [len(PERSIAN_EMBEDDINGS)]*20 if len(vec)==0 else [vec[0]]*(20-len(vec))+vec
    return torch.LongTensor(vec)

class LSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dims, num_layers, dropout, device, bidirectional=False):
        super(LSTM, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
        self.bidirectional = 2 if bidirectional else 1
        self.name = 'lstm_unit'

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_size=self.embedding_dim,
                            hidden_size=self.hidden_dims,
                            batch_first=True,
                            num_layers=self.num_layers,
                            bidirectional=bidirectional,
                            dropout = self.dropout)

        self.initialize_parameters()

    def initialize_parameters(self):
        """ Initializes network parameters. """

        state_dict = self.lstm.state_dict()
        for key in state_dict.keys():
            if 'weight' in key:
                hidden_dim = state_dict[key].size(0) / 4
                embed_dim = state_dict[key].size(1)
                # W
                if 'ih' in key:
                    state_dict[key] = Variable(torch.nn.init.normal_(state_dict[key],mean=0, std=.2) / torch.sqrt(torch.from_numpy(np.array(hidden_dim*embed_dim))))
                # U
                elif 'hh' in key:
                    state_dict[key] = Variable(torch.nn.init.normal_(state_dict[key], mean=0, std=.2) / torch.sqrt(torch.from_numpy(np.array(hidden_dim))))
            # b
            if 'bias' in key:
                hidden_dim = Variable((torch.tensor(state_dict[key].size(0) / 4).long()))
                # from paper
                state_dict[key] = Variable(torch.nn.init.uniform_(state_dict[key], a=-0.5, b=0.5))
                # paper says 2.5, their code has 1.5
                state_dict[key][hidden_dim:hidden_dim*2] = Variable(torch.tensor([2.5]))

        self.lstm.load_state_dict(state_dict)
        self.lstm = self.lstm.to(self.device)

    def init_hidden(self, batch_size):
        h = torch.zeros(self.bidirectional*self.num_layers, batch_size, self.hidden_dims).to(self.device)
        h = Variable(torch.nn.init.normal_(h, mean=0, std=.01).to(self.device))
        # h = Variable(torch.nn.init.xavier_normal_(h).to(self.device))

        c = torch.zeros(self.bidirectional*self.num_layers, batch_size, self.hidden_dims).to(self.device)
        c = Variable(torch.nn.init.normal_(c, mean=0, std=.01).to(self.device))
        # c = Variable(torch.nn.init.xavier_normal_(c).to(self.device))

        return h, c

    def forward(self, embeds, batch_size, hidden, cell):
        embeds = embeds.to(self.device) #.view(-1, batch_size, self.embedding_dim) #remove .view(...) if lstm batch_first is true

        output, (hidden,cell) = self.lstm(embeds, (hidden,cell))

        return output, hidden, cell


class SiameseLSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout, learning_rate, device):
        super(SiameseLSTM, self).__init__()

        self.char_embedding = nn.Embedding(num_embeddings=len(PERSIAN_EMBEDDINGS)+1, embedding_dim=embedding_dim).to(device) #remove if myword2vec is one-hot

        self.lstm = LSTM(embedding_dim, hidden_dim, num_layers, dropout, device)

        self.loss_fn = nn.CosineEmbeddingLoss(reduction='none')

        self.optimizer = optim.Adadelta(params=self.lstm.parameters(),
                                        lr=learning_rate,
                                        rho=0.95,
                                        eps=1e-6,
                                        weight_decay=0.0001)

    def forward(self, r1, r2, y):
        hidden_a_t, cell_a_t = self.lstm.init_hidden(r1.size(0))
        hidden_b_t, cell_b_t = self.lstm.init_hidden(r2.size(0))
        out_a, hidden_a_t, cell_a_t = self.lstm(embeds=self.char_embedding(r1),
                                                batch_size=r1.size(0),
                                                hidden=hidden_a_t,
                                                cell=cell_a_t)

        out_b, hidden_b_t, cell_b_t = self.lstm(embeds=self.char_embedding(r2),
                                                batch_size=r2.size(0),
                                                hidden=hidden_b_t,
                                                cell=cell_b_t)

        return (hidden_a_t.squeeze(), hidden_b_t.squeeze())

    def get_loss(self, e_pred, e, y):
        ''' Compute MSE between predictions and scaled gold labels '''
#         energy = self.loss_fn(e_pred, e)
#         loss = y*0.25*((1-energy)**2) + (1-y)*energy #use <m threshold for (1-y) case
        loss = self.loss_fn(e_pred, e, y)
        return loss

    def predict(self, r1, r2):
        with torch.no_grad():
            r1 = torch.unsqueeze(self.char_embedding(myword2vec(r1)), 0)
            r2 = torch.unsqueeze(self.char_embedding(myword2vec(r2)), 0)
            
            hidden_a_t, cell_a_t = self.lstm.init_hidden(r1.size(0))
            hidden_b_t, cell_b_t = self.lstm.init_hidden(r2.size(0))
            out_a, hidden_a_t, cell_a_t = self.lstm(embeds=r1,
                                                    batch_size=r1.size(0),
                                                    hidden=hidden_a_t,
                                                    cell=cell_a_t)

            out_b, hidden_b_t, cell_b_t = self.lstm(embeds=r2,
                                                    batch_size=r2.size(0),
                                                    hidden=hidden_b_t,
                                                    cell=cell_b_t)
            return nn.CosineSimilarity()(hidden_a_t.squeeze(dim=0), hidden_b_t.squeeze(dim=0))