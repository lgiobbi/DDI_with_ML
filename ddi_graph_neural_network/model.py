import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class Net(torch.nn.Module):
    """Graph Neural Network model for link prediction."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        """Initialize the GNN model.

        Args:
            in_channels (int): Number of input features.
            hidden_channels (int): Number of hidden features.
            out_channels (int): Number of output features.
        """
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Encode the input features using GCN layers.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Graph connectivity.

        Returns:
            torch.Tensor: Encoded node features.
        """
        x = self.conv1(x, edge_index)
        x = F.dropout(x, p=0.3)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.dropout(x, p=0.3)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return x

    def decode(self, z: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        """Decode edge labels from encoded node features.

        Args:
            z (torch.Tensor): Encoded node features.
            edge_label_index (torch.Tensor): Edge label indices.
        Returns:
            torch.Tensor: Decoded edge labels.
        """
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z: torch.Tensor) -> torch.Tensor:
        """Decode all node pairs.

        Args:
            z (torch.Tensor): Encoded node features.

        Returns:
            torch.Tensor: Decoded edge indices.
        """
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        """Forward pass for the GNN.

        Args:
            x (torch.Tensor): Input node features.
            edge_index (torch.Tensor): Graph connectivity.
            edge_label_index (torch.Tensor): Edge labels.

        Returns:
            torch.Tensor: Predicted edge labels.
        """
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)
