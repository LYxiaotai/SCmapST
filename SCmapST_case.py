import scanpy as sc
import numpy as np
import pandas as pd
import copy
import warnings 
import numpy as np
warnings.filterwarnings("ignore")
import random
import Map61 as cf1
import sys
import torch
from scipy import sparse
    
    
device = 'cuda:1'    
method = 'SCmapST'
savedir = '/home/L61/SCmapST/results/'
dataname = 'E1Z5'

sc_adata_file = savedir + 'E1Z5_sc.h5ad'
st_adata_file = savedir + 'E1Z5_st.h5ad'   
sc_adata_save_file = savedir  + dataname + '.h5ad'   

print('\n' + "ScmapST Model inference for " + dataname)

ad_st = sc.read_h5ad(st_adata_file)
ad_sc = sc.read_h5ad(sc_adata_file)

sc.pp.normalize_total(ad_sc)
sc.pp.log1p(ad_sc)
sc.pp.normalize_total(ad_st)
sc.pp.log1p(ad_st)    

df = pd.DataFrame(ad_st.obsm['spatial'],
            columns=['coord_x', 'coord_y'],
            index=ad_st.obs.index)
ad_st.obsm['spatial']= df
        
st_coor = np.array(ad_st.obsm['spatial'])  
st_gene = np.array(ad_st.X)
sc_coor = np.array(ad_sc.obsm['spatial'])
sc_gene = np.array(ad_sc.X)


cf1.pp_adatas(ad_sc, ad_st, genes=None)
training_genes = ad_sc.uns['training_genes']
gene_names = ad_sc.var_names  
selected_genes_idx = np.isin(gene_names, training_genes)
ad_sc = ad_sc[:, selected_genes_idx].copy()

training_genes = ad_st.uns['training_genes']
gene_names = ad_st.var_names  
selected_genes_idx = np.isin(gene_names, training_genes)
ad_st = ad_st[:, selected_genes_idx].copy()
    
    
if 'X_umap' not in ad_sc.obs:
    ad_sc.X = ad_sc.X.astype(np.float32)
    sc.pp.pca(ad_sc)
    sc.pp.neighbors(ad_sc)
    sc.tl.umap(ad_sc)


cf1.pp_adatas(ad_sc, ad_st, genes=None)
ad_map, rep1, rep2 = cf1.cell_space_map(ad_sc,
            ad_st,
            device='cuda:1',
            learning_rate=0.00001,
            num_epochs=2000)

mapping = ad_map.X      # numpy [sc,ST]

sc_rep = rep1[0].T.cpu().numpy()
st_rep = rep2[0].T.cpu().numpy()


print(method + 'writing...')
orisc = sc.read_h5ad(sc_adata_file)
orisc.uns[method + '_' + dataname + '_mapping_norm'] = sparse.csr_matrix(mapping)
orisc.uns[method + '_' + dataname + '_screp_norm'] = sc_rep
orisc.uns[method + '_' + dataname + '_strep_norm'] = st_rep
# orisc.obsm['ly_sc_coor_pred_norm'] = cf1.map_to_ST(sc_rep, st_rep, st_coor)     

sc.write(sc_adata_save_file, orisc)
            
            
