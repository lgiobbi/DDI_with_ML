# %% [markdown]
# # `DDI-LLM:`
# **Exploring Language-based Drug Chemical Structure Embedding Methods for Drug-Drug Interaction Prediction via Graph Convolutional Networks**
# ---
# `Drug-drug interactions (DDIs)` can arise when multiple drugs are used to treat complex or concurrent medical conditions, potentially leading to alterations in how these drugs work. Consequently, predicting DDIs has become a crucial endeavour within medical machine learning, addressing a critical aspect of healthcare.
# 
# This paper explores the application of language-based embeddings, including `BERT`, `GPT`,`LLaMA`, and `LLaMA2,` within the context of `Graph Convolutional Networks (GCN)` to enhance DDI prediction.
# 
# We start by  harnessing these advanced language models to generate embeddings for drug chemical structures and drug descriptions, providing a more comprehensive representation of drug characteristics. These embeddings are subsequently integrated into a DDI network, with GCN employed for link prediction. We utilize BERT, GPT, and LLaMA embeddings to improve the accuracy and effectiveness of predicting drug interactions within this network.
# 
# Our experiments reveal that using language-based drug embeddings in combination with DDI structure embeddings can yield accuracy levels comparable to state-of-the-art methods in DDI prediction.
# 
# 
# 
# 
# 

# %% [markdown]


# %% [markdown]
# 
# ## 1-   Setup
# 
# 
# 
# You need to have `Python >= 3.8` and install the following main packages:

# %%
%%capture
!pip install torch==2.0.0
!pip install torch_geometric
!pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
!pip install  deepchem
!pip install rdkit
!pip install gensim
!pip install git+https://github.com/samoturk/mol2vec
!pip install transformers


# %%
import pandas as pd
import numpy as np
import networkx as nx
import random
import os.path as osp
from rdkit import Chem
import deepchem as dc
from deepchem.feat.smiles_tokenizer import BasicSmilesTokenizer


# %%
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


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
import torch
from sklearn.metrics import roc_auc_score ,auc,precision_recall_curve,f1_score

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# %%
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


# %% [markdown]
# ## 2- Reading DDI dataset
# ---
# 
# This code read two data files, one for `DDI graph` and one for `drug info`. We take the intersection of the datasets, resulting a graph where each node has `SMILES`, `drug name`, and `drug description`.
# 
# 
# Input data:
# 
# * DDI network: It is a graph where nodes are drug IDs. There are 48514 edges, about `1514` unique `drugIDs`. Raw data from [`BioSnap`](https://snap.stanford.edu/biodata/datasets/10001/10001-ChCh-Miner.html)
# 
# Ref: [`MIRACLE`](https://arxiv.org/pdf/2010.11711.pdf)
# * `Drug ID-SMILES:` 11583 pairs of (DrugID, SMILES). [`DrugBank`](https://go.drugbank.com/releases/latest#structures)
# * `Drug ID-Desc:` 15235 pairs of (DrugID, SMILES) that we extracted from the DrugBANK database.
# 
# * **`NOTE:`** Certain SMILES strings in our drug dataset do not represent valid molecules, as determined by the RDKit library. Consequently, to maintain the accuracy and reliability of our graphs, we must exclude these drugs that lack valid information.

# %%


# %%
DDI_graph = pd.read_csv('https://raw.githubusercontent.com/liiniix/BioSNAP/master/ChCh-Miner/ChCh-Miner_durgbank-chem-chem.tsv', sep='\t')
# info about dataset: https://snap.stanford.edu/biodata/datasets/10001/10001-ChCh-Miner.html


# %%
G = nx.from_pandas_edgelist(DDI_graph, 'src', 'dst')



# %%
pos = nx.spring_layout(G)

nx.draw(G, pos, with_labels=False, node_size=3, node_color='skyblue')
nx.draw_networkx_edges(G, pos, alpha=0.5)

plt.title('Network Visualization')
plt.show()

# %% [markdown]
# `edge_index` is an important property that we will need for building GNNs. It is a list of edges with shape `[2, |E|]`. Important: since `ddi_graph` is undirected, $E$ includes both $(u, v)$ and $(v, u)$ for two drugs $u$ and $v$ that interact.
# 
# Note that there are no node features, so we will need to address this when building our model.

# %% [markdown]
# ### 2-2-Reading the SMILES strings
# ---

# %%
drugsSMILES = pd.read_csv('https://raw.githubusercontent.com/sshaghayeghs/molSMILES/main/structure%20links%202.csv')
drugID_smiles = drugsSMILES[["DrugBank ID", "SMILES"]]
drugID_smiles.dropna(inplace=True)
drugID_smiles.reset_index(drop=True, inplace=True)

# %%
drugsSMILES

# %% [markdown]
# ### 2-3- Reading Drug Description

# %%
drugsDESC = pd.read_csv('https://raw.githubusercontent.com/sshaghayeghs/molSMILES/main/Drug_description.csv')
drugID_DESC = drugsDESC[["Drug ID", "Discription"]]
drugID_DESC.dropna(inplace=True)
drugID_DESC.reset_index(drop=True, inplace=True)

# %%
drugsDESC
drugsDESC.to_csv('Drug_description.csv', index=False)

# %%
drugsDESC[:1000].to_csv('Drug_description_1000.csv', index=False)

# %% [markdown]
# Checking if the description reveals interaction information

# %%
drugsDESC

# %%
drugsDESC_masked = drugsDESC.copy()
drug_list = drugsDESC_masked['Drug Name'].to_list()
drug_ID_list = drugsDESC_masked['Drug ID'].to_list()
drugsDESC_masked

# %% [markdown]
# Scales bad with more entries, as lookup list and entries increases
# 
# 10000 entries -> 43s

# %%
"""
import re

# Compile regex ONCE for efficiency
pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, drug_list)) + r')\b')

def count_drugs_in_text(text):
    return len(pattern.findall(text))

def replace_drugs(text, replacement="<DRUG>"):
    return pattern.sub(replacement, text)

drugsDESC_masked['drug_count'] = drugsDESC_masked['Discription'].apply(count_drugs_in_text)
drugsDESC_masked['Description_Masked'] = drugsDESC_masked['Discription'].apply(replace_drugs)
"""

# %%
# Faster:

#!pip install flashtext

# find more synonyms for drugs here: https://go.drugbank.com/data_packages/drug_identifiers

# %%
from flashtext import KeywordProcessor

# Build processor
kp = KeywordProcessor(case_sensitive=False)
for drug in drug_list:
    kp.add_keyword(drug, "<DRUG>")
for drug_ID in drug_ID_list:
    kp.add_keyword(drug_ID, "<DRUG>")



# Replace & count
drugsDESC_masked['Description_Masked'] = drugsDESC_masked['Discription'].apply(kp.replace_keywords)
drugsDESC_masked['drug_count'] = drugsDESC_masked['Discription'].apply(
    lambda x: len(kp.extract_keywords(x))
)



# %%
drugsDESC_masked.sort_values('drug_count', ascending=False).head(10)

# %% [markdown]
# ### 2-4- Cleaning the DDI network
# Droping drugs with `invalid` smiles strings
# 
# * In our `drugID_smiles` dataset, some `SMILES` strings do not correspond to valid molecules according to `RDKit`. Our goal is to ensure the integrity of our ddi network by removing these non-valid drugs.
# 
# * Droping the drugs that do not have any correspond drug description.

# %%
#checking if a molecule has a valid molecule corespodn to the smiles string
def is_valid_molecule(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

# %%
valid_smiles = pd.DataFrame(drugID_smiles)
valid_smiles['IsValidMolecule'] = drugID_smiles['SMILES'].apply(is_valid_molecule)
df_valid_molecules = valid_smiles[valid_smiles['IsValidMolecule']]

# Drop the temporary 'IsValidMolecule' column
df_valid_molecules = df_valid_molecules.drop(columns=['IsValidMolecule'])

# %%

allowed_drug=[list(df_valid_molecules['DrugBank ID']),list(drugID_DESC['Drug ID'])]
# There are 1278 drugIDs that occur in the graph. Some graph nodes do not have associated SMILES or drug description

#droping the links that do not have any SMILES
for l in allowed_drug:
  for index, row in DDI_graph.iterrows():
      # Check if both cells in the row are in the allowed cells list
      if row['src'] not in l or row['dst'] not in l:
          # If either cell is not in the allowed cells list, remove the row
          DDI_graph.drop(index, inplace=True)



# %%
#27800 edges
DDI_graph=DDI_graph.reset_index(drop=True)


# %%
G = nx.from_pandas_edgelist(DDI_graph, 'src', 'dst')

# %%
# Find the connected components
connected_components = nx.connected_components(G)


num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
is_undirected = G.is_directed() is False
average_node_degree = sum(dict(G.degree()).values()) / num_nodes

has_isolated_nodes = any(deg == 0 for _, deg in G.degree())
has_self_loops = any(G.has_edge(n, n) for n in G.nodes())




# %%
#save the drugs smiles and drug description in the networks into a new dataframe
drugID_smiles_ddi = drugID_smiles[drugID_smiles['DrugBank ID'].isin(list(np.unique(DDI_graph.values)))]
drugID_smiles_ddi=drugID_smiles_ddi.reset_index(drop=True)
drugID_DESC_ddi = drugID_DESC[drugID_DESC['Drug ID'].isin(list(np.unique(DDI_graph.values)))]
drugID_DESC_ddi=drugID_DESC_ddi.reset_index(drop=True)



# %%
drugID_DESC_ddi

# %%
drugID_smiles_ddi

# %% [markdown]
# ## 3- Creating graph object for PyG
# ---
# A data object describing a homogeneous graph. The data object can hold node-level, link-level and graph-level attributes. In general, Data tries to mimic the behavior of a regular Python dictionary. In addition, it provides useful functionality for analyzing graph structures, and provides basic PyTorch tensor functionalities.
# [ref](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch_geometric.data.Data)

# %%
def PyG_data(feature,DDI_graph):
  DrugIDs_in_graph = np.unique(DDI_graph.values)
  node_id_map = {node_name: i for i, node_name in enumerate(DrugIDs_in_graph)}
  # Replace node names with integer IDs in the edge list
  src = [node_id_map[node_name] for node_name in DDI_graph['src']]
  dst = [node_id_map[node_name] for node_name in DDI_graph['dst']]
  # Stack the arrays side by side to create a 2D array
  combined_array = np.column_stack((np.array(src), np.array(dst)))
  edge_index = []  # List of tuples representing edges between drugs
  for drug_1, drug_2 in combined_array:
    # Create an undirected graph by adding edges in both directions
    edge_index.append((drug_1, drug_2))
    edge_index.append((drug_2, drug_1))
  #Replace node names with integer IDs in the feature
  feature=torch.tensor(feature,dtype=torch.float32)
  data = Data(x=feature, edge_index=torch.tensor(edge_index).t().contiguous())
  return data

# %%
transform = RandomLinkSplit(num_val=0.2,
    num_test=0.2,
    is_undirected=True,
    add_negative_train_samples=False,
    neg_sampling_ratio=1.0)
#train_data, val_data, test_data = transform(data)




# %% [markdown]
# ### 5-1- No Feature

# %%
def no_feature(smiles,DDI_graph):
  #DrugIDs_in_graph = np.unique(DDI_graph.values)
  features = np.ones((len(smiles),100))
  print('no_feature')
  return features,DDI_graph

# %% [markdown]
# ### 5-2- SMILES-based embedding

# %% [markdown]
# #### 5-2-1- Morgan Fingerprint

# %%
def Morgan(smiles,DDI_graph):
  featurizer = dc.feat.CircularFingerprint(size=100, radius=1)
  dataset=smiles['SMILES']
  features = pd.DataFrame(columns = [i for i in range(100)])
  for i in range(len(dataset)):
    features.loc[i] = featurizer.featurize(dataset[i])[0]
  print('Morgan')
  return features.values,DDI_graph

# %% [markdown]
# #### 5-2-2- Mol2vec
# 

# %%
def Mol2Vec(smiles,DDI_graph):
  featurizer = dc.feat.Mol2VecFingerprint()
  features=pd.DataFrame(columns = [i for i in range(300)])
  for s in smiles['SMILES']:
    features.loc[len(features)]=np.array(featurizer.featurize(s))[0]
  print('mol2vec')
  return features.values,DDI_graph

# %% [markdown]
# #### 5-2-3- SPVec

# %%
def sentences2vec(sentences, model,dim):
    keys = set(model.wv.index_to_key)
    vec = pd.DataFrame(columns = [i for i in range(dim)])
    for sentence in sentences:
            vec.loc[len(vec)] = np.array(sum([model.wv[y] for y in sentence         if y in set(sentence) & keys]))
    return vec

# %%
def character2vec(smiles,DDI_graph):
  tokenizer = BasicSmilesTokenizer()
  corpus=[]
  for s in smiles['SMILES']:
      corpus.append(tokenizer.tokenize(s))
  model= Word2Vec(corpus,
          vector_size=300,
          window=20,
          min_count=0,
          sg=1,
          epochs=5)
  aa_sentences = [tokenizer.tokenize(x) for x in smiles['SMILES']]
  vec=sentences2vec(aa_sentences, model,300)
  print('character2vec')
  return vec.values,DDI_graph

# %% [markdown]
# #### 5-2-4- Doc2vec
# ref:[gensim](https://radimrehurek.com/gensim/models/doc2vec.html)

# %%
def doc2vec(SMILES,DDI_graph):
  tagged_documents = [TaggedDocument(doc.split(), [i]) for i, doc in enumerate(SMILES['SMILES'])]

  model = Doc2Vec(tagged_documents, vector_size=100, min_count=1, epochs=20)

  feature = [model.infer_vector(doc.split()) for doc in SMILES['SMILES']]
  #feature=pd.DataFrame(embeddings,index=new_df['DrugBank ID'])
  print('doc2vec')
  return feature,DDI_graph

# %% [markdown]
# #### 5-2-5-Language models

# %%
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
    print(model_name)
    return  features.values, DDI_graph

# %% [markdown]
# ##6- GCN (Multi-view representation Fusion)
# ---
# 
# 1.   An encoder creates node embeddings by processing the graph with two
# convolution layers.
# 2.   We randomly add negative links to the original graph. This makes the model task a binary classification with the positive links from the original edges and the negative links from the added edges.
# 3.   A decoder makes link predictions (i.e. binary classifications) on all the edges including the negative links using node embeddings.# Models
# 
# 
# [Ref](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/link_pred.py)

# %% [markdown]
# This setup is from [the original link prediction implementation in Variational Graph Auto-Encoders](https://github.com/tkipf/gae). The code looks like something below. This is adapted from [the code example in PyG repo](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/link_pred.py) which is based on the Graph Auto-Encoders implementation.

# %% [markdown]
# The total number of the GCN encoder is 3.  To further regularise the model, dropout with 𝑝 = 0.3 is applied to every intermediate layer’s output.
# [MIRACLE](https://arxiv.org/pdf/2010.11711.pdf)

# %%
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
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(
            dim=-1
        )  # product of a pair of nodes on each edge

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

# %%
def train():
    model.train()
    optimizer.zero_grad()

    z = model.encode(train_data.x, train_data.edge_index) # initializing GCN model
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

# %% [markdown]
# ##7- Training and Evaluation
# ---

# %% [markdown]
# We can now train and evaluate the model with the following code.
# 
# 
# 

# %%
Embedding_models={#'No Feature':no_feature(drugID_smiles_ddi,DDI_graph),
                  #'Morgan':Morgan(drugID_smiles_ddi,DDI_graph),
                  #'Mol2vec':Mol2Vec(drugID_smiles_ddi,DDI_graph),
                  #'SPVec':character2vec(drugID_smiles_ddi,DDI_graph),
                  #'Doc2vec':doc2vec(drugID_smiles_ddi,DDI_graph),
                  'ChemBertaSMILES':LM(DDI_graph,allowed_drug,'Chemberta+SMILES','/content/drive/MyDrive/Shaghayegh Sadeghi/Drug embedding/DrugBank SMILES Embedding/SMILES_Chemberta.csv',','),
                  'MolformerSMILES':LM(DDI_graph,allowed_drug,'Molformer+SMILES','/content/drive/MyDrive/Shaghayegh Sadeghi/Drug embedding/DrugBank SMILES Embedding/SMILES_Molformer.csv',','),

                  #'SBERTSMILES':LM(DDI_graph,allowed_drug,'SBERT+SMILES','/content/drive/MyDrive/Shaghayegh Sadeghi/Drug embedding/DrugBank SMILES Embedding/SBERT/SMILES_SBert.csv',','),
                  #'AngledBERTSMILES':LM(DDI_graph,allowed_drug,'AngledBERT+SMILES','/content/drive/MyDrive/Shaghayegh Sadeghi/Drug embedding/DrugBank SMILES Embedding/AngleBERT/SMILES_angleBert.csv',','),
                  #'GPTSMILES':LM(DDI_graph,allowed_drug,'GPT+SMILES','/content/drive/MyDrive/Shaghayegh Sadeghi/Drug embedding/DrugBank SMILES Embedding/GPT/SMILES_GPT.csv','\t'),
                  #'LLaMASMILES':LM(DDI_graph,allowed_drug,'LLaMA+SMILES','/content/drive/MyDrive/Shaghayegh Sadeghi/Drug embedding/DrugBank SMILES Embedding/LLaMA/llama65b_base_SMILES_embeddings.csv','\t'),
                  #'LLaMA2SMILES':LM(DDI_graph,allowed_drug,'LLaMA2+SMILES','/content/drive/MyDrive/Shaghayegh Sadeghi/Drug embedding/DrugBank SMILES Embedding/LLaMA II/llamaII7b_base_SMILES_embeddings.csv','\t'),
                  #'AngledLLaMA2SMILES':LM(DDI_graph,allowed_drug,'AngledLLaMA+SMILES','/content/drive/MyDrive/Shaghayegh Sadeghi/Drug embedding/DrugBank SMILES Embedding/AngleLLaMA/SMILES_angleLlama.csv',','),
                  #'BERTDesc':LM(DDI_graph,allowed_drug,'BERT+Desc','/content/drive/MyDrive/Shaghayegh Sadeghi/Drug embedding/DrugBank Description Embedding/BERT/bert50mt_base_Discription_embeddings.csv','\t'),
                  #'SBERTDesc':LM(DDI_graph,allowed_drug,'SBERT+Desc','/content/drive/MyDrive/Shaghayegh Sadeghi/Drug embedding/DrugBank Description Embedding/SBERT/Desc_SBert.csv',','),
                  #'AngledBERTDesc':LM(DDI_graph,allowed_drug,'AngledBERT+Desc','/content/drive/MyDrive/Shaghayegh Sadeghi/Drug embedding/DrugBank Description Embedding/AngleBERT/Drug_description_angleBERT.csv',','),
                  #'GPTDesc':LM(DDI_graph,allowed_drug,'GPT+Desc','/content/drive/MyDrive/Shaghayegh Sadeghi/Drug embedding/DrugBank Description Embedding/GPT/Dr_Desc_GPT.csv','\t'),
                  #'LLaMADesc':LM(DDI_graph,allowed_drug,'LLaMA+Desc','/content/drive/MyDrive/Shaghayegh Sadeghi/Drug embedding/DrugBank Description Embedding/LLaMA/llama65b_base_Discription_embeddings.csv','\t'),
                  #'LLaMA2Desc':LM(DDI_graph,allowed_drug,'LLaMA2+Desc','/content/drive/MyDrive/Shaghayegh Sadeghi/Drug embedding/DrugBank Description Embedding/LLaMA II/llamaII7b_base_Discription_embeddings.csv','\t'),
                  #'AngledLLaMA2Desc':LM(DDI_graph,allowed_drug,'AngledLLaMA+Desc','/content/drive/MyDrive/Shaghayegh Sadeghi/Drug embedding/DrugBank Description Embedding/AngleLLaMA/Drug_description_angleLlama.csv',','),
                  }

# %% [markdown]
# We set each parameter group’s learning rate using an exponentially decaying schedule with the initial learning rate 0.0001
# and multiplicative factor 0.96. [MIRACLE](https://arxiv.org/pdf/2010.11711.pdf)

# %%
lmbda = lambda epoch: 0.96

# %%
LR=[#0.01,0.001,
    0.0001
    #,0.0002,0.0003,0.00001
    ]


#modelname=['No Feature','Morgan','Mol2vec','SPVec','Doc2Vec',
         # 'BERTSMILES','SBERTSMILES','AngledBERTSMILES','GPTSMILES','LLaMASMILES','LLaMA2SMILES','AngledLLaMA2SMILES',
         # 'BERTDesc','SBERTDesc','AngledBERTDesc','GPTDesc','LLaMADesc','LLaMA2Desc','AngledLLaMA2Desc']


modelname=['BERTSMILES','BERTSMILES_token']
AUC=pd.DataFrame(columns = [#'0.01','0.001',
                            '0.0001'
                            #,'0.0002','0.0003','0.00001'
                            ])
PR=pd.DataFrame(columns = [#'0.01','0.001',
                            '0.0001'
                            #,'0.0002','0.0003','0.00001'
                            ])


AUC['Embedding']=modelname
PR['Embedding']=modelname

for l in LR:
  print('-------------------------------')
  print('=====Learning Rate:',l,'=======')
  print('-------------------------------')
  results_AUC=[]
  results_PR=[]
  for modelname, emb in Embedding_models.items():
    print('-------------------------------')
    print('=========',modelname,'=========')
    print('-------------------------------')
    data=PyG_data(emb[0],emb[1])
    train_data, val_data, test_data = transform(data)
    model = Net(data.num_features, 256, 256).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=l)
    scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)
    criterion = torch.nn.BCEWithLogitsLoss()
    '''
    neg_edge_index = negative_sampling(
          edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
          num_neg_samples=train_data.edge_label_index.size(1), method='sparse')'''
    struct_neg_tup=structured_negative_sampling(edge_index=train_data.edge_index,num_nodes=train_data.num_nodes,contains_neg_self_loops = False)
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
    for epoch in range(1, 100):
        loss = train()
        val_auc = test(val_data)[0]
        test_auc = test(test_data)[0]
        label=test(test_data)[1]
        score=test(test_data)[2]
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            final_test_auc = test_auc
            best_scores=score
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}')

    precision, recall, thresholds = precision_recall_curve(label, best_scores)
    pr=auc(recall, precision)
    results_AUC.append({"Embedding":modelname,"AUC":final_test_auc})
    results_PR.append({"Embedding":modelname,"PR_AUC": pr})
    del data
    del model

  #AUC[str(l)]=results_AUC['AUC']
  #PR[str(l)]=results_PR['PR_AUC']

# %%
results_AUC

# %%
results_PR

# %% [markdown]


# %% [markdown]
# ## 8- Results

# %%
PR.to_csv('/content/drive/MyDrive/Shaghayegh Sadeghi/DDI_LM/PR_BioSnap_Struc_v2.csv')

# %%
AUC.to_csv('/content/drive/MyDrive/Shaghayegh Sadeghi/DDI_LM/AUC_BioSnap_Struc_v2.csv')


# %%
AUC

# %%
AUC=pd.read_csv('/content/drive/MyDrive/Shaghayegh Sadeghi/DDI_LM/AUC_BioSnap_Struc_v2.csv',index_col=[0])

AUC.index=AUC['Embedding']
AUC_edited=AUC.drop(['0.00001','Embedding'],axis=1)
model=['GPTSMILES','LLaMASMILES','LLaMA2SMILES']
AUC_edited_f= AUC_edited[AUC_edited.index.isin(model)]
AUC_edited_f.index=['GPT','LLaMA','LLaMA2']

# %%
AUC_edited_f

# %%
PR=pd.read_csv('/content/drive/MyDrive/Shaghayegh Sadeghi/DDI_LM/PR_BioSnap_Struc_v2.csv',index_col=[0])
PR.index=AUC['Embedding']
PR_edited=PR.drop(['0.00001','Embedding'],axis=1)
PR_edited_f= PR_edited[PR_edited.index.isin(model)]
PR_edited_f.index=['GPT','LLaMA','LLaMA2']

# %%
import matplotlib.pyplot as plt

# Disable LaTeX rendering
plt.rcParams['text.usetex'] = False

# %%
plt.clf()


# %%
df = pd.DataFrame(AUC_edited_f)
df

# %%
df = pd.DataFrame(AUC_edited_f)


box_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd','#fced47']

plt.figure(figsize=(10, 6))
bp = plt.boxplot(df.values, patch_artist=True, boxprops=dict(facecolor='white', edgecolor='black'))

for box, color in zip(bp['boxes'], box_colors):
    box.set(color='black', linewidth=2)
    box.set(facecolor=color)

for whisker in bp['whiskers']:
    whisker.set(color='black', linestyle='--', linewidth=1)

for median in bp['medians']:
    median.set(color='black', linewidth=2)

plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.title('AUC Distribution by Learning Rate', fontsize=16)
plt.xlabel('Learning Rate', fontsize=12)
plt.ylabel('AUC', fontsize=12)

plt.xticks(range(1, len(df.columns) + 1), df.columns, fontsize=10)


plt.tight_layout()
plt.savefig("AUC Distribution by Learning Rate_biosnap.pdf", format="pdf", bbox_inches="tight")

plt.show()




# %%
df = pd.DataFrame(PR_edited_f)


box_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd','#fced47']


plt.figure(figsize=(10, 6))
bp = plt.boxplot(df.values, patch_artist=True, boxprops=dict(facecolor='white', edgecolor='black'))


for box, color in zip(bp['boxes'], box_colors):
    box.set(color='black', linewidth=2)
    box.set(facecolor=color)


for whisker in bp['whiskers']:
    whisker.set(color='black', linestyle='--', linewidth=1)

for median in bp['medians']:
    median.set(color='black', linewidth=2)


plt.grid(axis='y', linestyle='--', alpha=0.5)


plt.title('AUPR Distribution by Learning Rate', fontsize=16)
plt.xlabel('Learning Rate', fontsize=12)
plt.ylabel('AUPR', fontsize=12)

plt.xticks(range(1, len(df.columns) + 1), df.columns, fontsize=10)


plt.tight_layout()
plt.savefig("AUPR Distribution by Learning Rate_biosnap.pdf", format="pdf", bbox_inches="tight")

plt.show()

# %%
sns.set_style("whitegrid")
sns.set_palette("Set1")

# %%
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=AUC_edited_f.index, y='0.0001', data=AUC_edited_f)
sns.set_style("whitegrid")
sns.set_palette("Set1")
plt.title('BioSnap - AUROC', fontsize=16)
plt.xlabel('Embedding', fontsize=12)
plt.ylabel('AUROC', fontsize=12)
plt.ylim(0, 1)

ax.set_xticklabels(ax.get_xticklabels(), ha='right', fontsize=10)

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("BioSnap - AUROC.pdf", format="pdf", bbox_inches="tight")

plt.show()

# %%
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=PR_edited_f.index, y='0.0001', data=PR_edited_f)


plt.title('BioSnap - AUPR', fontsize=16)
plt.xlabel('Embedding', fontsize=12)
plt.ylabel('AUPR', fontsize=12)
plt.ylim(0, 1)

ax.set_xticklabels(ax.get_xticklabels(), ha='right', fontsize=10)

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')

#ax.spines['top'].set_visible(False)
#ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("BioSnap - AUPR.pdf", format="pdf", bbox_inches="tight")

plt.show()







# %%

# Create a Seaborn line plot with multiple lines for each column
plt.figure(figsize=(10, 6))

# Iterate through additional columns and plot them as separate lines
for col in AUC_edited_f.columns[1:]:
    ax = sns.lineplot(data=AUC_edited_f, x=AUC_edited_f.index, y=col, marker='o', label=col, linewidth=2)

# Customize the plot
plt.title('AUC', fontsize=16)
plt.xlabel('Learning Rate', fontsize=12)
plt.ylabel('AUC', fontsize=12)
plt.ylim(0, 1)  # Adjust the y-axis limits as needed

ax.set_xticklabels(AUC_edited.index, rotation=45, ha='right', fontsize=10)

# Customize tick label font size
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Add a legend
plt.legend(title='Learning Rate', loc='upper right')

# Add a grid
plt.grid(True, linestyle='--', alpha=0.5)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("BioSnap - AUC-line.pdf", format="pdf", bbox_inches="tight")

plt.show()



# %%

# Create a Seaborn line plot with multiple lines for each column
plt.figure(figsize=(10, 6))

# Iterate through additional columns and plot them as separate lines
for col in PR_edited.columns[1:]:
    ax = sns.lineplot(data=PR_edited, x='Embedding', y=col, marker='o', label=col, linewidth=2)

# Customize the plot
plt.title('AUPR', fontsize=16)
plt.xlabel('Learning Rate', fontsize=12)
plt.ylabel('AUPR', fontsize=12)
plt.ylim(0, 1)  # Adjust the y-axis limits as needed

ax.set_xticklabels(PR_edited.index, rotation=45, ha='right', fontsize=10)

# Customize tick label font size
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Add a legend
plt.legend(title='Learning Rate', loc='upper right')

# Add a grid
plt.grid(True, linestyle='--', alpha=0.5)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("BioSnap - AUPR-line.pdf", format="pdf", bbox_inches="tight")

plt.show()




