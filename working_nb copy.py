
import pandas as pd
import numpy as np
import networkx as nx
import random
import os.path as osp
from rdkit import Chem
import deepchem as dc
from deepchem.feat.smiles_tokenizer import BasicSmilesTokenizer

import matplotlib.pyplot as plt
import seaborn as sns

from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
import torch_geometric.transforms as T
from torch_geometric.utils import negative_sampling, convert, to_dense_adj,structured_negative_sampling
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn.conv import MessagePassing

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR,MultiplicativeLR
from torch import Tensor
from torch.utils.data import DataLoader

from gensim.models import word2vec
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pickle

from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec
from sklearn.metrics import roc_auc_score ,auc,precision_recall_curve,f1_score

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

DDI_graph = pd.read_csv('https://raw.githubusercontent.com/liiniix/BioSNAP/master/ChCh-Miner/ChCh-Miner_durgbank-chem-chem.tsv', sep='\t')
G = nx.from_pandas_edgelist(DDI_graph, 'src', 'dst')

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=False, node_size=3, node_color='skyblue')
nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.show()

drugsSMILES = pd.read_csv('https://raw.githubusercontent.com/sshaghayeghs/molSMILES/main/structure%20links%202.csv')
drugID_smiles = drugsSMILES[["DrugBank ID", "SMILES"]]
drugID_smiles.dropna(inplace=True)
drugID_smiles.reset_index(drop=True, inplace=True)

drugsDESC = pd.read_csv('https://raw.githubusercontent.com/sshaghayeghs/molSMILES/main/Drug_description.csv')
drugID_DESC = drugsDESC[["Drug ID", "Discription"]]
drugID_DESC.dropna(inplace=True)
drugID_DESC.reset_index(drop=True, inplace=True)

drugsDESC_masked = drugsDESC.copy()
drug_list = drugsDESC_masked['Drug Name'].to_list()
drug_ID_list = drugsDESC_masked['Drug ID'].to_list()

from flashtext import KeywordProcessor
kp = KeywordProcessor(case_sensitive=False)
for drug in drug_list:
    kp.add_keyword(drug, "<DRUG>")
for drug_ID in drug_ID_list:
    kp.add_keyword(drug_ID, "<DRUG>")

drugsDESC_masked['Description_Masked'] = drugsDESC_masked['Discription'].apply(kp.replace_keywords)
drugsDESC_masked['drug_count'] = drugsDESC_masked['Discription'].apply(
    lambda x: len(kp.extract_keywords(x))
)

def is_valid_molecule(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

valid_smiles = pd.DataFrame(drugID_smiles)
valid_smiles['IsValidMolecule'] = drugID_smiles['SMILES'].apply(is_valid_molecule)
df_valid_molecules = valid_smiles[valid_smiles['IsValidMolecule']]
df_valid_molecules = df_valid_molecules.drop(columns=['IsValidMolecule'])

allowed_drug=[list(df_valid_molecules['DrugBank ID']),list(drugID_DESC['Drug ID'])]

for l in allowed_drug:
  for index, row in DDI_graph.iterrows():
      if row['src'] not in l or row['dst'] not in l:
          DDI_graph.drop(index, inplace=True)

DDI_graph=DDI_graph.reset_index(drop=True)
G = nx.from_pandas_edgelist(DDI_graph, 'src', 'dst')
connected_components = nx.connected_components(G)

drugID_smiles_ddi = drugID_smiles[drugID_smiles['DrugBank ID'].isin(list(np.unique(DDI_graph.values)))]
drugID_smiles_ddi=drugID_smiles_ddi.reset_index(drop=True)
drugID_DESC_ddi = drugID_DESC[drugID_DESC['Drug ID'].isin(list(np.unique(DDI_graph.values)))]
drugID_DESC_ddi=drugID_DESC_ddi.reset_index(drop=True)

def PyG_data(feature,DDI_graph):
  DrugIDs_in_graph = np.unique(DDI_graph.values)
  node_id_map = {node_name: i for i, node_name in enumerate(DrugIDs_in_graph)}
  src = [node_id_map[node_name] for node_name in DDI_graph['src']]
  dst = [node_id_map[node_name] for node_name in DDI_graph['dst']]
  combined_array = np.column_stack((np.array(src), np.array(dst)))
  edge_index = []
  for drug_1, drug_2 in combined_array:
    edge_index.append((drug_1, drug_2))
    edge_index.append((drug_2, drug_1))
  feature=torch.tensor(feature,dtype=torch.float32)
  data = Data(x=feature, edge_index=torch.tensor(edge_index).t().contiguous())
  return data

transform = RandomLinkSplit(num_val=0.2,
    num_test=0.2,
    is_undirected=True,
    add_negative_train_samples=False,
    neg_sampling_ratio=1.0)

def no_feature(smiles,DDI_graph):
  features = np.ones((len(smiles),100))
  return features,DDI_graph

def Morgan(smiles,DDI_graph):
  featurizer = dc.feat.CircularFingerprint(size=100, radius=1)
  dataset=smiles['SMILES']
  features = pd.DataFrame(columns = [i for i in range(100)])
  for i in range(len(dataset)):
    features.loc[i] = featurizer.featurize(dataset[i])[0]
  return features.values,DDI_graph

def Mol2Vec(smiles,DDI_graph):
  featurizer = dc.feat.Mol2VecFingerprint()
  features=pd.DataFrame(columns = [i for i in range(300)])
  for s in smiles['SMILES']:
    features.loc[len(features)]=np.array(featurizer.featurize(s))[0]
  return features.values,DDI_graph

def sentences2vec(sentences, model,dim):
    keys = set(model.wv.index_to_key)
    vec = pd.DataFrame(columns = [i for i in range(dim)])
    for sentence in sentences:
            vec.loc[len(vec)] = np.array(sum([model.wv[y] for y in sentence if y in set(sentence) & keys]))
    return vec

def character2vec(smiles,DDI_graph):
  tokenizer = BasicSmilesTokenizer()
  corpus=[]
  for s in smiles['SMILES']:
      corpus.append(tokenizer.tokenize(s))
  model= Word2Vec(corpus, vector_size=300, window=20, min_count=0, sg=1, epochs=5)
  aa_sentences = [tokenizer.tokenize(x) for x in smiles['SMILES']]
  vec=sentences2vec(aa_sentences, model,300)
  return vec.values,DDI_graph

def doc2vec(SMILES,DDI_graph):
  tagged_documents = [TaggedDocument(doc.split(), [i]) for i, doc in enumerate(SMILES['SMILES'])]
  model = Doc2Vec(tagged_documents, vector_size=100, min_count=1, epochs=20)
  feature = [model.infer_vector(doc.split()) for doc in SMILES['SMILES']]
  return feature,DDI_graph

allowed_drug=list(df_valid_molecules['DrugBank ID'])+list(drugID_DESC['Drug ID'])
def LM(DDI_graph,allowed_drug,model_name,dir,s):
    Drug=pd.read_csv(dir, sep=s,index_col=0)
    if 'Unnamed: 0' in Drug.columns:
      Drug.drop(columns='Unnamed: 0', inplace=True)
    df = Drug[Drug.iloc[:, 0].isin(allowed_drug)]
    df=df.reset_index(drop=True)
    if 'Discription' in df.columns:
      features=df.drop(df.columns[[0, 1, 2]], axis=1)
    else:
      features=df.drop(df.columns[[0, 1]], axis=1)
    return  features.values, DDI_graph

class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
    def encode(self, x, edge_index):
        x=self.conv1(x, edge_index)
        x=F.dropout(x, p=0.3)
        x=F.relu(x)
        x=self.conv2(x, edge_index)
        x=F.dropout(x, p=0.3)
        x=F.relu(x)
        x=self.conv3(x, edge_index)
        return x
    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)
    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

def train():
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
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    roc=roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())
    label=data.edge_label.cpu().numpy()
    score=out.cpu().numpy()
    return roc,label,score