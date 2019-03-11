import torch
import torch.nn as nn
import torch.nn.functional as F

class CharCNN(nn.Module):
    """CharCNN model"""
    def __init__(self, num_classes: int, embedding_dim: int, dic: dict) -> None:
        """Instantiating CharCNN

        Args:
            num_classes: number of classes
            embedding_dim: embedding dimension of token
            dic: token2idx
        """
        super(CharCNN, self).__init__()
        self.embedding = nn.Embedding(len(dic), embedding_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=256, kernel_size=7)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=7)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3)
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3)
        self.conv5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3)
        self.conv6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3)

        self.fc1 = nn.Linear(in_features=1792, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=num_classes)

        self.apply(self.__init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_batch = self.embedding(x)
        x_batch = x_batch.permute(0, 2, 1)
        fmap = F.max_pool1d(F.relu(self.conv1(x_batch)), 3, 3)
        fmap = F.max_pool1d(F.relu(self.conv2(fmap)), 3, 3)
        fmap = F.relu(self.conv3(fmap))
        fmap = F.relu(self.conv4(fmap))
        fmap = F.relu(self.conv5(fmap))
        fmap = F.max_pool1d(F.relu(self.conv6(fmap)), 3, 3)
        flatten = fmap.view(fmap.shape[0], -1)
        dense = F.dropout(F.relu(self.fc1(flatten)))
        dense = F.dropout(F.relu(self.fc2(dense)))
        score = F.dropout(self.fc3(dense))
        return score

    def __init_weights(self, layer) -> None:
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_uniform_(layer.weight)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
