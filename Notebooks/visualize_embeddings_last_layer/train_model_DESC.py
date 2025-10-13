# %%
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import structured_negative_sampling
from torch_geometric.nn import GCNConv
from torch.optim.lr_scheduler import MultiplicativeLR
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def PyG_data(feature, DDI_graph):
    DrugIDs_in_graph = np.unique(DDI_graph.values)
    node_id_map = {node_name: i for i, node_name in enumerate(DrugIDs_in_graph)}
    src = [node_id_map[node_name] for node_name in DDI_graph['src']]
    dst = [node_id_map[node_name] for node_name in DDI_graph['dst']]
    combined_array = np.column_stack((np.array(src), np.array(dst)))
    edge_index = []
    for drug_1, drug_2 in combined_array:
        edge_index.append((drug_1, drug_2))
        edge_index.append((drug_2, drug_1))
    feature = torch.tensor(feature, dtype=torch.float32)
    data = Data(x=feature, edge_index=torch.tensor(edge_index).t().contiguous())
    return data

def LM(DDI_graph, allowed_drug, dir, sep='\t'):
    Drug = pd.read_csv(dir, sep=sep, index_col=0)
    if 'Unnamed: 0' in Drug.columns:
        Drug.drop(columns='Unnamed: 0', inplace=True)
    df = Drug[Drug.iloc[:, 0].isin(allowed_drug)].reset_index(drop=True)
    if 'Discription' in df.columns:
        features = df.drop(df.columns[[0, 1, 2]], axis=1)
    else:
        features = df.drop(df.columns[[0, 1]], axis=1)
    return features.values, DDI_graph

class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.dropout(x, p=0.3)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.dropout(x, p=0.3)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)

def train(model, optimizer, criterion, scheduler, train_data, edge_label_index, edge_label):
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    out = model.decode(z, edge_label_index).view(-1)
    loss = criterion(out, edge_label)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss

@torch.no_grad()
def test(model, data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    roc = roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())
    label = data.edge_label.cpu().numpy()
    score = out.cpu().numpy()
    return roc, label, score

def no_feature(smiles, DDI_graph):
    features = np.ones((len(smiles), 100))
    print('no_feature')
    return features, DDI_graph

def lmbda(epoch):
    return 0.96


# Train a single model and return model, train/val/test data, and metrics
def run_training(emb, transform, device, lmbda, epochs=100, patience=10, lr=0.0003):
    print('-------------------------------')
    print(f'Training with LR: {lr}')
    data = PyG_data(emb[0], emb[1])
    train_data, val_data, test_data = transform(data)
    model = Net(data.num_features, 256, 256).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)
    criterion = torch.nn.BCEWithLogitsLoss()

    struct_neg_tup = structured_negative_sampling(
        edge_index=train_data.edge_index,
        num_nodes=train_data.num_nodes,
        contains_neg_self_loops=False
    )
    neg_edge_index = torch.stack((struct_neg_tup[0], struct_neg_tup[2]), dim=0)
    neg_edge_index, _ = torch.unique(neg_edge_index, dim=1, return_inverse=True)

    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat([
        train_data.edge_label,
        train_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)

    best_val_auc = final_test_auc = 0
    wait = 0
    best_model_state = None
    for epoch in range(1, epochs):
        loss = train(model, optimizer, criterion, scheduler, train_data, edge_label_index, edge_label)
        val_auc, _, _ = test(model, val_data)
        test_auc, label, score = test(model, test_data)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            final_test_auc = test_auc
            best_scores = score
            wait = 0
            best_model_state = model.state_dict()
        else:
            wait += 1
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}')
        if wait >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    precision, recall, _ = precision_recall_curve(label, best_scores)
    pr = auc(recall, precision)
    metrics = {"AUC": final_test_auc, "PR_AUC": pr}
    return model, data, metrics


def main():
    models = {'GPT+Desc': '/data/giobbi/embeddings/Dr_Desc_GPT.csv'}    

    # Data loading
    DDI_graph = pd.read_csv('https://raw.githubusercontent.com/liiniix/BioSNAP/master/ChCh-Miner/ChCh-Miner_durgbank-chem-chem.tsv', sep='\t')
    DDI_graph.rename(columns={'Drug1': 'src', 'Drug2': 'dst'}, inplace=True)
    
    transform = RandomLinkSplit(
        num_val=0.2,
        num_test=0.2,
        is_undirected=True,
        add_negative_train_samples=False,
        neg_sampling_ratio=1.0
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 100
    LR = [0.0003]

    results = {}
    for modelname, dir in models.items():
        current_graph = DDI_graph

        drugID_DESC = pd.read_csv(dir, sep='\t', index_col=0)
        allowed_drug = list(drugID_DESC['Drug ID'])

        emb = LM(current_graph, allowed_drug, dir)

        for lr in LR:
            print(f'======== {modelname} | LR: {lr} ========')
            model, data, metrics = run_training(emb, transform, device, lmbda, epochs=epochs, lr=lr)
            results[(modelname, lr)] = {'model': model, 'data': data, 'metrics': metrics}
    
    for key, value in results.items():
        print(f"Model: {key[0]}, LR: {key[1]}, Metrics: {value['metrics']}")

if __name__ == "__main__":
    main()

