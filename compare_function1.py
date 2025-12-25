import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.neighbors import KDTree
from collections import Counter
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm
import torch
import random
from scipy.stats import spearmanr


import numpy as np
import pandas as pd
import scanpy as sc
import torch
import logging
import copy
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
#import utils as ut
from pathlib import Path
import argparse
import random
import matplotlib.cm as cm
import torch.nn as nn
from torch.autograd import Variable
import os
import torch.multiprocessing
from tqdm import tqdm
import time
from sklearn import preprocessing
from sklearn.decomposition import NMF
from sklearn.decomposition import FastICA
from torch.nn.functional import softmax, cosine_similarity
logging.getLogger().setLevel(logging.INFO)


from typing import List, Optional

import numpy as np
import torch
from sklearn.utils.extmath import randomized_svd
from torch import Tensor


def dual_pca(
    device,
    X: np.ndarray,
    Y: np.ndarray,
    dim: Optional[int] = 50,
    singular: Optional[bool] = False,
    backend: Optional[str] = "sklearn"
    ) -> List[Tensor]:
    r"""
    Dual PCA for batch correction

    Parameters
    ----------
    X
        expr matrix 1 in shape of (cells, genes)
    Y
        expr matrix 2 in shape of (cells, genes)
    dim
        dimension of embedding
        assert Dim_X > Dim_Y
    singular
        if multiply the singular value
    backend
        backend to calculate singular value
    use_gpu
        if calculate in gpu

    Returns
    ----------
    embd1, embd2: Tensors of embedding

    References
    ----------
    Thanks Xin-Ming Tu for his [blog](https://xinmingtu.cn/blog/2022/CCA_dual_PCA/)
    """
    assert X.shape[1] == Y.shape[1]
    X = torch.Tensor(X).to(device=device)
    Y = torch.Tensor(Y).to(device=device)
    cor_var = X @ Y.T
    if backend == "torch":
        U, S, Vh = torch.linalg.svd(cor_var)
        if not singular:
            return U[:, :dim], Vh.T[:, :dim]
        Z_x = U[:, :dim] @ torch.sqrt(torch.diag(S[:dim]))
        Z_y = Vh.T[:, :dim] @ torch.sqrt(torch.diag(S[:dim]))
        return Z_x.cpu(), Z_y.cpu()
        # torch.dist(cor_var, Z_x @ Z_y.T)  # check the information loss
    elif backend == "sklearn":
        cor_var = cor_var.cpu().numpy()
        U, S, Vh = randomized_svd(cor_var, n_components=dim, random_state=0)
        if not singular:
            return Tensor(U), Tensor(Vh.T)
        Z_x = U @ np.sqrt(np.diag(S))
        Z_y = Vh.T @ np.sqrt(np.diag(S))
        return Tensor(Z_x), Tensor(Z_y)


def annotate_gene_sparsity(adata):
    """
    Annotates gene sparsity in given Anndatas. 
    Update given Anndata by creating `var` "sparsity" field with gene_sparsity (1 - % non-zero observations).
    Args:
        adata (Anndata): single cell or spatial data.
    Returns:
        None
    """
    mask = adata.X != 0
    gene_sparsity = np.sum(mask, axis=0) / adata.n_obs
    gene_sparsity = np.asarray(gene_sparsity)
    gene_sparsity = 1 - np.reshape(gene_sparsity, (-1,))
    adata.var["sparsity"] = gene_sparsity
    
    
def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    
def pp_adatas(adata_sc, adata_sp, genes=None, gene_to_lowercase = True):
    """
    Pre-process AnnDatas so that they can be mapped. Specifically:
    - Remove genes that all entries are zero
    - Find the intersection between adata_sc, adata_sp and given marker gene list, save the intersected markers in two adatas
    - Calculate density priors and save it with adata_sp
    Args:
        adata_sc (AnnData): single cell data
        adata_sp (AnnData): spatial expression data
        genes (List): Optional. List of genes to use. If `None`, all genes are used.
    
    Returns:
        update adata_sc by creating `uns` `training_genes` `overlap_genes` fields 
        update adata_sp by creating `uns` `training_genes` `overlap_genes` fields and creating `obs` `rna_count_based_density` & `uniform_density` field
    """

    # remove all-zero-valued genes
    sc.pp.filter_genes(adata_sc, min_cells=1)
    sc.pp.filter_genes(adata_sp, min_cells=1)

    if genes is None:
        # Use all genes
        genes = adata_sc.var.index
               
    # put all var index to lower case to align
    if gene_to_lowercase:
        adata_sc.var.index = [g.lower() for g in adata_sc.var.index]
        adata_sp.var.index = [g.lower() for g in adata_sp.var.index]
        genes = list(g.lower() for g in genes)

    adata_sc.var_names_make_unique()
    adata_sp.var_names_make_unique()
    

    # Refine `marker_genes` so that they are shared by both adatas
    genes = list(set(genes) & set(adata_sc.var.index) & set(adata_sp.var.index))
    # logging.info(f"{len(genes)} shared marker genes.")

    adata_sc.uns["training_genes"] = genes
    adata_sp.uns["training_genes"] = genes
    logging.info(
        "{} training genes are saved in `uns``training_genes` of both single cell and spatial Anndatas.".format(
            len(genes)
        )
    )

    # Find overlap genes between two AnnDatas
    overlap_genes = list(set(adata_sc.var.index) & set(adata_sp.var.index))
    # logging.info(f"{len(overlap_genes)} shared genes.")

    adata_sc.uns["overlap_genes"] = overlap_genes
    adata_sp.uns["overlap_genes"] = overlap_genes
    logging.info(
        "{} overlapped genes are saved in `uns``overlap_genes` of both single cell and spatial Anndatas.".format(
            len(overlap_genes)
        )
    )

    # Calculate uniform density prior as 1/number_of_spots
    adata_sp.obs["uniform_density"] = np.ones(adata_sp.X.shape[0]) / adata_sp.X.shape[0]
    logging.info(
        f"uniform based density prior is calculated and saved in `obs``uniform_density` of the spatial Anndata."
    )

    # Calculate rna_count_based density prior as % of rna molecule count
    rna_count_per_spot = np.array(adata_sp.X.sum(axis=1)).squeeze()
    adata_sp.obs["rna_count_based_density"] = rna_count_per_spot / np.sum(rna_count_per_spot)
    logging.info(
        f"rna count based density prior is calculated and saved in `obs``rna_count_based_density` of the spatial Anndata."
    )


def get_data(
    adata_sc,
    adata_sp,
    device='cuda:1',
    learning_rate=0.00001,
    num_epochs=2000,
    ):
    seed_value = 2023
    seed_everything(seed_value)
    if not set(["training_genes", "overlap_genes"]).issubset(set(adata_sc.uns.keys())):
        raise ValueError("Missing tangram parameters. Run `pp_adatas()`.")

    if not set(["training_genes", "overlap_genes"]).issubset(set(adata_sp.uns.keys())):
        raise ValueError("Missing tangram parameters. Run `pp_adatas()`.")

    assert list(adata_sp.uns["training_genes"]) == list(adata_sc.uns["training_genes"])

    training_genes = adata_sc.uns["training_genes"]
    if isinstance(adata_sc.X, csc_matrix) or isinstance(adata_sc.X, csr_matrix):
        S = np.array(adata_sc[:, training_genes].X.toarray(), dtype="float32",)
    elif isinstance(adata_sc.X, np.ndarray):
        S = np.array(adata_sc[:, training_genes].X.toarray(), dtype="float32",)
    else:
        X_type = type(adata_sc.X)
        logging.error("AnnData X has unrecognized type: {}".format(X_type))
        raise NotImplementedError

    if isinstance(adata_sp.X, csc_matrix) or isinstance(adata_sp.X, csr_matrix):
        G = np.array(adata_sp[:, training_genes].X.toarray(), dtype="float32")
    elif isinstance(adata_sp.X, np.ndarray):
        G = np.array(adata_sp[:, training_genes].X, dtype="float32")
    else:
        X_type = type(adata_sp.X)
        logging.error("AnnData X has unrecognized type: {}".format(X_type))
        raise NotImplementedError

    if not S.any(axis=0).all() or not G.any(axis=0).all():
        raise ValueError("Genes with all zero values detected. Run `pp_adatas()`.")

    sc_LC = adata_sc.obsm["X_umap"]
    sc_LC = np.array(sc_LC).astype(np.float32)
    location0_sc=((sc_LC[:,0]).astype(np.float32))
    location1_sc=((sc_LC[:,1]).astype(np.float32))

    xmax0 = max(location0_sc)
    xmin0 = min(location0_sc)
    xmax1 = max(location1_sc)
    xmin1 = min(location1_sc)

    len0=xmax0-xmin0
    len1=xmax1-xmin1
    sc_location=np.array([len0,len1])

    sp_LC = adata_sp.obsm["spatial"]
    sp_LC = np.array(sp_LC).astype(np.float32)
    location0_st = (sp_LC[:,0]).astype(np.float32)
    location1_st = (sp_LC[:,1]).astype(np.float32)

    xmax0 = max(location0_st)
    xmin0 = min(location0_st)
    xmax1 = max(location1_st)
    xmin1 = min(location1_st)

    len0=xmax0-xmin0
    len1=xmax1-xmin1
    st_location=np.array([len0,len1])

    sc_LC = sc_LC.reshape((1, 1, sc_LC.shape[0],2))
    sp_LC = sp_LC.reshape((1, 1, sp_LC.shape[0],2))

    S = S.astype(np.float32)
    #S = np.transpose(S)
    G = G.astype(np.float32)
    #G = np.transpose(G)
    #Snew = preprocessing.scale(S)
    #Gnew = preprocessing.scale(G)

    Snew = S.reshape((S.shape[0],1,S.shape[1]))
    Gnew = G.reshape((G.shape[0],1,G.shape[1]))

    Snew = np.transpose(Snew)
    Gnew = np.transpose(Gnew)

    Snew, Gnew = torch.from_numpy(Snew), torch.from_numpy(Gnew)
    
    #logT = preprocessing.FunctionTransformer(np.log1p)
    
    scaler1 = preprocessing.StandardScaler().fit(S)
    Snew_scale = scaler1.transform(S)
    scaler2 = preprocessing.StandardScaler().fit(G)
    Gnew_scale = scaler2.transform(G)
    
    #pca = FastICA(n_components=100,random_state=0,whiten='unit-variance')

    #Snew_scale = pca.fit_transform(Snew_scale)
    #Gnew_scale = pca.fit_transform(Gnew_scale)

    Snew_scale = Snew_scale.astype(np.float32)
    Gnew_scale = Gnew_scale.astype(np.float32)

    Snew_scale = Snew_scale.reshape((Snew_scale.shape[0],1,Snew_scale.shape[1]))
    Gnew_scale = Gnew_scale.reshape((Gnew_scale.shape[0],1,Gnew_scale.shape[1]))
    
    #Snew_scale=copy.deepcopy(S)
    #Gnew_scale=copy.deepcopy(G)

    #Snew_scale = Snew_scale.reshape((S.shape[0],1,S.shape[1]))
    #Gnew_scale = Gnew_scale.reshape((G.shape[0],1,G.shape[1]))
    
    Snew_scale = np.transpose(Snew_scale)
    Gnew_scale = np.transpose(Gnew_scale)

    Snew_scale, Gnew_scale = torch.from_numpy(Snew_scale), torch.from_numpy(Gnew_scale)
    #print(Snew_scale.shape)

    kpsc_np, kpst_np = torch.from_numpy(sc_LC), torch.from_numpy(sp_LC)
    sc_location, st_location = torch.from_numpy(sc_location), torch.from_numpy(st_location)
    #seed = 1
    #seed_everything(seed)
    
    zx, zy = dual_pca(device, S, G)
    zx = zx.reshape((1,zx.shape[0],zx.shape[1]))
    zy = zy.reshape((1,zy.shape[0],zy.shape[1]))


    data={  'keypoints0': list(kpsc_np),
            'keypoints1': list(kpst_np),
            'emb0': zx,
            'emb1': zy,
            'descrip0Scale': list(Snew_scale),
            'descrip1Scale': list(Gnew_scale),
            'descriptors0': list(Snew),
            'descriptors1': list(Gnew),
            'sc_location': (sc_location),
            'st_location': (st_location),
        }
    
    kpt1 = data['keypoints0'][0]
    size1 = data['sc_location'].reshape(1,2)
    desc1_0 = torch.stack(data['descrip0Scale'], dim=-1)
    desc1 = data['emb0']
    descriptors1 = torch.stack(data['descriptors0'])
    kpt2 = data['keypoints1'][0]
    size2 = data['st_location'].reshape(1,2)
    desc2_0 = torch.stack(data['descrip1Scale'], dim=-1)
    desc2 = data['emb1']
    descriptors2 = torch.stack(data['descriptors1'])

    test_data={'x1':kpt1,'x2':kpt2,
            'desc1':desc1,'desc2':desc2,
            'size1':size1,'size2':size2,
            'descriptors1':descriptors1, 'descriptors2':descriptors2
            }
    
    return test_data
    




def cell_space_map(
    adata_sc,
    adata_sp,
    device='cuda:1',
    learning_rate=0.00001,
    num_epochs=2000,
    ):
    seed_value = 2023
    seed_everything(seed_value)
    if not set(["training_genes", "overlap_genes"]).issubset(set(adata_sc.uns.keys())):
        raise ValueError("Missing tangram parameters. Run `pp_adatas()`.")

    if not set(["training_genes", "overlap_genes"]).issubset(set(adata_sp.uns.keys())):
        raise ValueError("Missing tangram parameters. Run `pp_adatas()`.")

    assert list(adata_sp.uns["training_genes"]) == list(adata_sc.uns["training_genes"])

    training_genes = adata_sc.uns["training_genes"]
    if isinstance(adata_sc.X, csc_matrix) or isinstance(adata_sc.X, csr_matrix):
        S = np.array(adata_sc[:, training_genes].X.toarray(), dtype="float32",)
    elif isinstance(adata_sc.X, np.ndarray):
        S = np.array(adata_sc[:, training_genes].X.toarray(), dtype="float32",)
    else:
        X_type = type(adata_sc.X)
        logging.error("AnnData X has unrecognized type: {}".format(X_type))
        raise NotImplementedError

    if isinstance(adata_sp.X, csc_matrix) or isinstance(adata_sp.X, csr_matrix):
        G = np.array(adata_sp[:, training_genes].X.toarray(), dtype="float32")
    elif isinstance(adata_sp.X, np.ndarray):
        G = np.array(adata_sp[:, training_genes].X, dtype="float32")
    else:
        X_type = type(adata_sp.X)
        logging.error("AnnData X has unrecognized type: {}".format(X_type))
        raise NotImplementedError

    if not S.any(axis=0).all() or not G.any(axis=0).all():
        raise ValueError("Genes with all zero values detected. Run `pp_adatas()`.")

    sc_LC = adata_sc.obsm["X_umap"]
    sc_LC = np.array(sc_LC).astype(np.float32)
    location0_sc=((sc_LC[:,0]).astype(np.float32))
    location1_sc=((sc_LC[:,1]).astype(np.float32))

    xmax0 = max(location0_sc)
    xmin0 = min(location0_sc)
    xmax1 = max(location1_sc)
    xmin1 = min(location1_sc)

    len0=xmax0-xmin0
    len1=xmax1-xmin1
    sc_location=np.array([len0,len1])

    sp_LC = adata_sp.obsm["spatial"]
    sp_LC = np.array(sp_LC).astype(np.float32)
    location0_st = (sp_LC[:,0]).astype(np.float32)
    location1_st = (sp_LC[:,1]).astype(np.float32)

    xmax0 = max(location0_st)
    xmin0 = min(location0_st)
    xmax1 = max(location1_st)
    xmin1 = min(location1_st)

    len0=xmax0-xmin0
    len1=xmax1-xmin1
    st_location=np.array([len0,len1])

    sc_LC = sc_LC.reshape((1, 1, sc_LC.shape[0],2))
    sp_LC = sp_LC.reshape((1, 1, sp_LC.shape[0],2))

    S = S.astype(np.float32)
    #S = np.transpose(S)
    G = G.astype(np.float32)
    #G = np.transpose(G)
    #Snew = preprocessing.scale(S)
    #Gnew = preprocessing.scale(G)

    Snew = S.reshape((S.shape[0],1,S.shape[1]))
    Gnew = G.reshape((G.shape[0],1,G.shape[1]))

    Snew = np.transpose(Snew)
    Gnew = np.transpose(Gnew)

    Snew, Gnew = torch.from_numpy(Snew), torch.from_numpy(Gnew)
    
    #logT = preprocessing.FunctionTransformer(np.log1p)
    
    scaler1 = preprocessing.StandardScaler().fit(S)
    Snew_scale = scaler1.transform(S)
    scaler2 = preprocessing.StandardScaler().fit(G)
    Gnew_scale = scaler2.transform(G)
    
    #pca = FastICA(n_components=100,random_state=0,whiten='unit-variance')

    #Snew_scale = pca.fit_transform(Snew_scale)
    #Gnew_scale = pca.fit_transform(Gnew_scale)

    Snew_scale = Snew_scale.astype(np.float32)
    Gnew_scale = Gnew_scale.astype(np.float32)

    Snew_scale = Snew_scale.reshape((Snew_scale.shape[0],1,Snew_scale.shape[1]))
    Gnew_scale = Gnew_scale.reshape((Gnew_scale.shape[0],1,Gnew_scale.shape[1]))
    
    #Snew_scale=copy.deepcopy(S)
    #Gnew_scale=copy.deepcopy(G)

    #Snew_scale = Snew_scale.reshape((S.shape[0],1,S.shape[1]))
    #Gnew_scale = Gnew_scale.reshape((G.shape[0],1,G.shape[1]))
    
    Snew_scale = np.transpose(Snew_scale)
    Gnew_scale = np.transpose(Gnew_scale)

    Snew_scale, Gnew_scale = torch.from_numpy(Snew_scale), torch.from_numpy(Gnew_scale)
    #print(Snew_scale.shape)

    kpsc_np, kpst_np = torch.from_numpy(sc_LC), torch.from_numpy(sp_LC)
    sc_location, st_location = torch.from_numpy(sc_location), torch.from_numpy(st_location)
    #seed = 1
    #seed_everything(seed)
    
    zx, zy = dual_pca(device, S, G)
    zx = zx.reshape((1,zx.shape[0],zx.shape[1]))
    zy = zy.reshape((1,zy.shape[0],zy.shape[1]))


    data={  'keypoints0': list(kpsc_np),
            'keypoints1': list(kpst_np),
            'emb0': zx,
            'emb1': zy,
            'descrip0Scale': list(Snew_scale),
            'descrip1Scale': list(Gnew_scale),
            'descriptors0': list(Snew),
            'descriptors1': list(Gnew),
            'sc_location': (sc_location),
            'st_location': (st_location),
        }
    kpt1 = data['keypoints0'][0]
    size1 = data['sc_location'].reshape(1,2)
    desc1_0 = torch.stack(data['descrip0Scale'], dim=-1)
    desc1 = data['emb0']
    descriptors1 = torch.stack(data['descriptors0'])
    kpt2 = data['keypoints1'][0]
    size2 = data['st_location'].reshape(1,2)
    desc2_0 = torch.stack(data['descrip1Scale'], dim=-1)
    desc2 = data['emb1']
    descriptors2 = torch.stack(data['descriptors1'])

    test_data={'x1':kpt1,'x2':kpt2,
            'desc1':desc1,'desc2':desc2,
            'size1':size1,'size2':size2,
            'descriptors1':descriptors1, 'descriptors2':descriptors2
            }
    
    config = {
            'ma': {
                'descriptor_dim': desc1.shape[2],
                'device': device,
            }
        }
    
    from match_model import matcher as SGM_Model
    mapper = SGM_Model(config.get('ma', {}))
    mapper.to(config.get('ma', {})['device'])
    optimizer = torch.optim.Adam(mapper.parameters(), lr=learning_rate)
    
    print("Start training")
    for epoch in range(1, num_epochs+1):
        t = time.time()
        epoch_loss = 0
        mapper.train()
        for k in test_data:
            test_data[k] = Variable(test_data[k].to(config.get('ma', {})['device']))
        
        pred,_,_ = mapper(test_data)     # pred['p'][:, :-1, :-1].shape: [1,st,sc]
    
        Loss,score0,score1 = mapper.get_loss(pred,test_data)
        optimizer.zero_grad()        
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=False, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

        Loss.backward()
        torch.nn.utils.clip_grad_norm_(mapper.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(Loss)
        
        t1=time.time()-t
        # if epoch%100==0:
        if epoch%100==0:
            print ('Epoch {}, Loss: {:.3f},Score0: {:.3f},Score1: {:.3f},Time: {:.3f}' 
                .format(epoch, Loss.item(),score0.item(),score1.item(),t1)) 
    
    with torch.no_grad():
        pred, rep1, rep2 = mapper(test_data)
        mapping_matrix = softmax(pred[0,:,:], dim=-1).cpu().numpy()
        #F_out = torch.sigmoid(F).cpu().numpy()

    training_genes = adata_sc.uns["training_genes"]
    adata_map = sc.AnnData(
        X=mapping_matrix,
        obs=adata_sc[:, training_genes].obs.copy(),
        var=adata_sp[:, training_genes].obs.copy(),
    )
    G_predicted = adata_map.X.T @ S
    cos_sims = []
    for v1, v2 in zip(G.T, G_predicted.T):
        norm_sq = np.linalg.norm(v1) * np.linalg.norm(v2)
        cos_sims.append((v1 @ v2) / norm_sq)

    df_cs = pd.DataFrame(cos_sims, training_genes, columns=["train_score"])
    df_cs = df_cs.sort_values(by="train_score", ascending=False)
    adata_map.uns["train_genes_df"] = df_cs

    # Annotate sparsity of each training genes
    annotate_gene_sparsity(adata_sc)
    annotate_gene_sparsity(adata_sp)
    adata_map.uns["train_genes_df"]["sparsity_sc"] = adata_sc[
        :, training_genes
    ].var.sparsity
    adata_map.uns["train_genes_df"]["sparsity_sp"] = adata_sp[
        :, training_genes
    ].var.sparsity
    adata_map.uns["train_genes_df"]["sparsity_diff"] = (
        adata_sp[:, training_genes].var.sparsity
        - adata_sc[:, training_genes].var.sparsity
    )
    
    return adata_map, rep1, rep2




def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy import sparse
import pandas as pd
from sklearn.neighbors import KDTree
from collections import Counter
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm
from scipy.stats import spearmanr
import torch
    

def getModelSimMat(rep_x, rep_y):
    
    similarity_matrix = cosine_similarity(rep_x, rep_y)
    sorted_similarity_matrix = []
    index_matrix = []

    for i in range(similarity_matrix.shape[0]):
        sorted_idx = np.argsort(similarity_matrix[i,:])[::-1]
        sorted_row = similarity_matrix[i,sorted_idx]
        sorted_similarity_matrix.append(sorted_row)
        index_matrix.append(sorted_idx)
        
    sorted_similarity_matrix = np.asarray(sorted_similarity_matrix)
    index_matrix = np.asarray(index_matrix)

    return sorted_similarity_matrix,index_matrix,similarity_matrix


def map_to_ST(query_rep,ref_rep,ref_coors):
    
    sorted_similarity_matrix,index_matrix,similarity_matrix =  getModelSimMat(query_rep,ref_rep)
    
    predicted_coors = []
    
    for i, ind in enumerate(index_matrix):
        
        # ith query data
        # ind: index of reference data
        cur_predict_st = ind[0]
        cur_st_coor = ref_coors[cur_predict_st]
        predicted_coors.append(cur_st_coor)
    
    predicted_coors = np.asarray(predicted_coors)
    
    return predicted_coors

