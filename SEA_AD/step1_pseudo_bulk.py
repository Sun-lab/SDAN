
import scanpy as sc
import numpy as np
import anndata as ad
import pandas as pd
import glob
import re

# --------------------------------------------------
# list file names
# --------------------------------------------------

work_dir = '/Users/wsun/research/data/SEA-AD/'
d = '/Users/wsun/research/data/SEA-AD/*.h5ad'

files = glob.glob(d)
len(files)

# --------------------------------------------------
# read in data
# --------------------------------------------------

for f1 in files:
    print(f1)
    cell_type = re.findall("SEA-AD/(\S+).h5ad", f1)[0]
    cell_type

    adata = ad.read_h5ad(f1)
    adata.obs

    list(adata.obs.columns.values)
    adata.obs.donor_id.value_counts()
    adata.obs.cell_type.value_counts()
    adata.obs.disease.value_counts()

    adata.var.iloc[0:5,1:3]

    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False,
                               inplace=True, use_raw=True)

    list(adata.obs.columns.values)
    list(adata.var.columns.values)

    adata.var.to_csv(work_dir+cell_type+'_genes_info.csv')
    adata.obs.to_csv(work_dir+cell_type+'_cell_info.csv')

    edat = adata.raw.X.toarray()
    print(edat.shape)
    print(edat[5:8, 13:18])

    adata.obs.index.is_unique
    df = pd.DataFrame(data=edat, index=adata.obs.index, columns=adata.var.index)
    df['donor_id'] = adata.obs['donor_id']
    pseudo_bulk = df.groupby("donor_id").sum()
    pseudo_bulk.to_csv(work_dir+cell_type+'_pseudo_bulk.csv')
    
