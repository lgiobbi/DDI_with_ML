# %%
import pandas as pd
import numpy as np
from rdkit import Chem

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


# Utility Functions
def is_valid_molecule(smiles) -> bool:
    """Check if a SMILES string corresponds to a valid molecule."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False

def get_valid_smiles(drugID_smiles):
    valid_smiles = pd.DataFrame(drugID_smiles)
    valid_smiles['IsValidMolecule'] = drugID_smiles['SMILES'].apply(is_valid_molecule)
    df_valid_molecules = valid_smiles[valid_smiles['IsValidMolecule']]
    return df_valid_molecules.drop(columns=['IsValidMolecule'])


# Flexible DDI_graph filtering: filter by 'smiles' or 'desc'
def filter_ddi_graph(DDI_graph, allowed_df, filter_type):
    if filter_type == 'smiles':
        allowed_drug = set(allowed_df['DrugBank ID'])
    elif filter_type == 'desc':
        allowed_drug = set(allowed_df['Drug ID'])
    else:
        raise ValueError("filter_type must be 'smiles' or 'desc'")
    return DDI_graph[DDI_graph['src'].isin(allowed_drug) & DDI_graph['dst'].isin(allowed_drug)].reset_index(drop=True)

def get_ddi_drug_info(drugID_smiles, drugID_DESC, DDI_graph):
    unique_ids = list(np.unique(DDI_graph.values))
    drugID_smiles_ddi = drugID_smiles[drugID_smiles['DrugBank ID'].isin(unique_ids)].reset_index(drop=True)
    drugID_DESC_ddi = drugID_DESC[drugID_DESC['Drug ID'].isin(unique_ids)].reset_index(drop=True)
    return drugID_smiles_ddi, drugID_DESC_ddi

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

def LM(DDI_graph, allowed_drug, model_name, dir, sep):
    Drug = pd.read_csv(dir, sep=sep, index_col=0)
    if 'Unnamed: 0' in Drug.columns:
        Drug.drop(columns='Unnamed: 0', inplace=True)
    df = Drug[Drug.iloc[:, 0].isin(allowed_drug)].reset_index(drop=True)
    if 'Discription' in df.columns:
        features = df.drop(df.columns[[0, 1, 2]], axis=1)
    else:
        features = df.drop(df.columns[[0, 1]], axis=1)
    print(model_name)
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

def run_training(Embedding_models, transform, device, lmbda, epochs=100, patience=10, LR=[0.0003]):
    modelnames = [name for name in Embedding_models.keys()]
    AUC = pd.DataFrame({'Embedding': modelnames})
    PR = pd.DataFrame({'Embedding': modelnames})

    for lr in LR:
        print('-------------------------------')
        print(f'=====Learning Rate: {lr} =======')
        print('-------------------------------')
        results_AUC = []
        results_PR = []
        for modelname, emb in Embedding_models.items():
            print('-------------------------------')
            print(f'========= {modelname} =========')
            print('-------------------------------')
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
                # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}')
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            if best_model_state is not None:
                model.load_state_dict(best_model_state)

            precision, recall, _ = precision_recall_curve(label, best_scores)
            pr = auc(recall, precision)
            results_AUC.append({"Embedding": emb, "AUC": final_test_auc})
            results_PR.append({"Embedding": emb, "PR_AUC": pr})

        AUC[str(lr)] = [r["AUC"] for r in results_AUC]
        PR[str(lr)] = [r["PR_AUC"] for r in results_PR]

    desired_order = ['Embedding'] + [str(lr) for lr in LR]
    AUC = AUC[desired_order]
    PR = PR[desired_order]
    return AUC, PR

def main():
    # Data loading
    DDI_graph = pd.read_csv('https://raw.githubusercontent.com/liiniix/BioSNAP/master/ChCh-Miner/ChCh-Miner_durgbank-chem-chem.tsv', sep='\t')
    DDI_graph.rename(columns={'Drug1': 'src', 'Drug2': 'dst'}, inplace=True)
    #drugsSMILES = pd.read_csv('https://raw.githubusercontent.com/sshaghayeghs/molSMILES/main/structure%20links%202.csv')
    #drugID_smiles = drugsSMILES[["DrugBank ID", "SMILES"]]
    #drugID_smiles.dropna(inplace=True)
    #drugID_smiles.reset_index(drop=True, inplace=True)
    drugsDESC = pd.read_csv('https://raw.githubusercontent.com/sshaghayeghs/molSMILES/main/Drug_description.csv')
    drugID_DESC = drugsDESC[["Drug ID", "Discription"]]
    drugID_DESC.dropna(inplace=True)
    drugID_DESC.reset_index(drop=True, inplace=True)

    #df_valid_molecules = get_valid_smiles(drugID_smiles)

    # Example: filter by SMILES first, then by description
    #DDI_graph = filter_ddi_graph(DDI_graph, df_valid_molecules, 'smiles')
    DDI_graph = filter_ddi_graph(DDI_graph, drugID_DESC, 'desc')


    #drugID_smiles_ddi, drugID_DESC_ddi = get_ddi_drug_info(drugID_smiles, drugID_DESC, DDI_graph)

    #allowed_drug = list(df_valid_molecules['DrugBank ID']) + list(drugID_DESC['Drug ID'])

    allowed_drug = list(drugID_DESC['Drug ID'])
    Embedding_models = {
        'GPTDesc': LM(DDI_graph, allowed_drug, 'GPT+Desc', '/data/giobbi/embeddings/Dr_Desc_GPT.csv', '\t'),
        # Add other models as needed
    }

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
    AUC, PR = run_training(Embedding_models, transform, device, lmbda, epochs=epochs, LR=LR)
    print(AUC)
    print(PR)

if __name__ == "__main__":
    main()


