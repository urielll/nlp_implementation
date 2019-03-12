import torch
import torch.nn as nn
import torch.nn.functional as F
from gluonnlp import Vocab

class SentenceCNN(nn.Module):
    """SentenceCNN class"""
    def __init__(self, num_classes: int, vocab: Vocab) -> None:
        """Instantiating SentenceCNN class

        Args:
            num_classes (int): the number of classes
            vocab (gluonnlp.Vocab): the instance of gluonnlp.Vocab
        """
        super(SentenceCNN, self).__init__()
        # static embedding
        self.static = nn.Embedding(len(vocab), embedding_dim=300, padding_idx=0)
        self.static.weight.data.copy_(torch.from_numpy(vocab.embedding.idx_to_vec.asnumpy()))
        self.static.weight.requires_grad_(False)

        # non-static embedding
        self.non_static = nn.Embedding(len(vocab), embedding_dim=300, padding_idx=0)
        self.non_static.weight.data.copy_(torch.from_numpy(vocab.embedding.idx_to_vec.asnumpy()))

        # convolution layer
        self.tri_gram = nn.Conv1d(in_channels=300, out_channels=100, kernel_size=3)
        self.tetra_gram = nn.Conv1d(in_channels=300, out_channels=100, kernel_size=4)
        self.penta_gram = nn.Conv1d(in_channels=300, out_channels=100, kernel_size=5)

        # output layer
        self.fc = nn.Linear(in_features=300, out_features=num_classes)

        # dropout
        self.drop = nn.Dropout()

        # initialization
        self.apply(self.__init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # embedding layer
        static_batch = self.static(x)
        static_batch = static_batch.permute(0, 2, 1) # for Conv1d

        non_static_batch = self.non_static(x)
        non_static_batch = non_static_batch.permute(0, 2, 1) # for Conv1d

        # convolution layer (extract feature)
        tri_feature = F.relu(self.tri_gram(static_batch)) + F.relu(self.tri_gram(non_static_batch))
        tetra_feature = F.relu(self.tetra_gram(static_batch)) + F.relu(self.tetra_gram(non_static_batch))
        penta_feature = F.relu(self.penta_gram(static_batch)) + F.relu(self.penta_gram(non_static_batch))

        # max-overtime pooling
        tri_feature = torch.max(tri_feature, 2)[0]
        tetra_feature = torch.max(tetra_feature, 2)[0]
        penta_feature = torch.max(penta_feature, 2)[0]
        feature = torch.cat((tri_feature, tetra_feature, penta_feature), 1)

        # dropout
        feature = self.drop(feature)

        # output layer
        score = self.fc(feature)

        return score

    def __init_weights(self, layer) -> None:
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_uniform_(layer.weight)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)