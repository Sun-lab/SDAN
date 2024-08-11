import anndata

adata = anndata.read_h5ad("SEA_AD/output/test_reduced_Astro.h5ad")
print(adata.shape)
print(adata.obs)

data_all = anndata.read_h5ad("/Users/wsun/research/data/SEA-AD/Astro.h5ad")
print(data_all.shape)
print(data_all.obs)
