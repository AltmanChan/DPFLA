from torch import nn


class BiLSTM(nn.Module):
    def __init__(self, num_words, embedding_dim = 100, dropout = 0.25):
        super(BiLSTM, self).__init__()
        """
        Given embedding_matrix: numpy array with vector for all words
        return prediction ( in torch tensor format)
        """
        self.embedding = nn.Embedding(
                                      num_embeddings=num_words+1,
                                      embedding_dim=embedding_dim)
        # LSTM with hidden_size = 128
        self.lstm = nn.LSTM(
                            embedding_dim,
                            128,
                            bidirectional=True,
                            batch_first=True,
                             )
        # Input(512) because we use bi-directional LSTM ==> hidden_size*2 + maxpooling **2  = 128*4 = 512,
        #will be explained more on forward method
        self.out = nn.Linear(512, 1)
    def forward(self, x):
        # pass input (tokens) through embedding layer
        x = self.embedding(x)
        # fit embedding to LSTM
        self.lstm.flatten_parameters()
        hidden, _ = self.lstm(x)
        # apply mean and max pooling on lstm output
        avg_pool= torch.mean(hidden, 1)
        max_pool, index_max_pool = torch.max(hidden, 1)
        # concat avg_pool and max_pool ( so we have 256 size, also because this is bidirectional ==> 256*2 = 512)
        out = torch.cat((avg_pool, max_pool), 1)
        # fit out to self.out to conduct dimensionality reduction from 512 to 1
        out = self.out(out)
        # return output
        return out