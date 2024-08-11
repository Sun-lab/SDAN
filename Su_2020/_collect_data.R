
library(data.table)
library(readxl)
library(Matrix)
library(tidyr)
library(ggplot2)
library(viridis)
library(ggpointdensity)
library(ggpubr)

theme_set(theme_classic())

# cell_type = "cd8"

args = commandArgs(trailingOnly=TRUE)
args

for(i in 1:length(args)){
  eval(parse(text=args[[i]]))
}

cell_type

data_dir = "../../aTCR_large_files/Su_2020"

# ------------------------------------------------------------------------
# read in file information
# ------------------------------------------------------------------------

finfo = fread("ArrayExpress/file_info.csv")
dim(finfo)
table(finfo$disease)

finfo = finfo[which(finfo$disease == "COVID-19"),]
dim(finfo)
table(finfo$disease)

finfo[1:2,]

# ------------------------------------------------------------------------
# only keep the file at baseline
# ------------------------------------------------------------------------

table(finfo$sampling_time_point)

finfo = finfo[which(finfo$sampling_time_point == "BL"),]
dim(finfo)
table(finfo$gex_dir)
finfo$gex_file[1:5]

# ------------------------------------------------------------------------
# extract gene expression data
# ------------------------------------------------------------------------

gex_dat = gex_sub = NULL
cell_info = NULL

genes2check = c("CD3E", "CD3D", "CD3G", "CD8A", "GZMK", "CD4", 
                "CD19", "CD79A", "IGLV2-8", "IGHV1-2", "IGHV2-5")

for(i in 1:nrow(finfo)){
  if(i %% 10 == 0){ cat(i, date(), "\n") }
  
  dir_ct  = paste0(cell_type, "_tcr_dir")
  file_ct = paste0(cell_type, "_tcr_file")
  
  tcr_dir  = gsub(".zip", "", finfo[[dir_ct]][i], fixed = TRUE)
  tcr_file = finfo[[file_ct]][i]
  tcr_dat  = fread(file.path(data_dir, tcr_dir, tcr_file))
  
  dim(tcr_dat)
  tcr_dat[1:2,]
  
  cell = tcr_dat$V1
  
  gex_dir  = gsub(".zip", "", finfo$gex_dir[i], fixed = TRUE)
  gex_file = finfo$gex_file[i]
  gex = fread(file.path(data_dir, gex_dir, gex_file))
  dim(gex)
  gex[1:3,1:2]
  
  stopifnot(all(cell %in% gex$V1))

  tcr_dat[,':='(individual=finfo$individual[i], 
                time_point=finfo$sampling_time_point[i])]
  
  cell_info = rbind(cell_info, 
                    tcr_dat[,.(V1, chain_pairing, clonotype, 
                               clonotype_size, clonal_expansion,
                               individual, time_point)])
  
  gex_idx = match(cell, gex$V1)
  
  if(i == 1){
    gex_names = names(gex)
  }else{
    stopifnot(all(gex_names == names(gex)))
  }
  
  
  gex_i = gex[gex_idx,]
  gex_i[,V1:=NULL]
  
  stopifnot(all(genes2check %in% names(gex_i)))
  gex_sub = rbind(gex_sub, data.matrix(gex_i[,..genes2check]))
  gex_i = as(data.matrix(gex_i), "sparseMatrix")
  
  dim(gex_i)
  gex_i[1:3,1:2]
  
  gex_dat = rbind(gex_dat, gex_i)
}

gex_sub = data.frame(gex_sub)
dim(gex_sub)
gex_sub[1:2,]

apply(gex_sub, 2, quantile, probs=c(0.5, 0.75, 0.95, 0.99, 0.999))

dim(gex_dat)
gex_dat[1:2,1:4]

prop0 = colSums(gex_dat > 0)/nrow(gex_dat)
length(prop0)
summary(prop0)
table(prop0 > 0.02)

# ------------------------------------------------------------------------
# convert gene expression to count data
# ------------------------------------------------------------------------

min_y = apply(gex_dat, 1, function(xx){ min(xx[xx > 0])})
summary(min_y)
d = exp(min_y) - 1
summary(d)

gex_count = (exp(gex_dat) - 1)/d

max_dev = apply(abs(gex_count - round(gex_count)), 1, max)
summary(max_dev)
gex_count = round(gex_count)

summary(colSums(gex_count))
w2kp = which(colSums(gex_count) > 5000)
gex_dat[1:5,w2kp[1:10]]
gex_count[1:5,w2kp[1:10]]

rm(gex_dat)

# ------------------------------------------------------------------------
# check mito genes
# ------------------------------------------------------------------------

mito_genes = fread("../Annotation/mito_genes.tsv")
dim(mito_genes)
mito_genes[1:2,]

table(mito_genes$hgnc_symbol %in% colnames(gex_count))

gex_mito = gex_count[,colnames(gex_count) %in% mito_genes$hgnc_symbol]
dim(gex_mito)
gex_mito[1:2,]
apply(gex_mito, 2, quantile, probs=c(0.5, 0.75, 0.95, 0.99, 0.999))

mito_total = rowSums(gex_mito)
summary(mito_total)

total = rowSums(gex_count)
summary(total)

q13 = quantile(mito_total, probs=c(0.25, 0.75))
total_cutoff = q13[2] + 1.5*(q13[2] - q13[1])
total_cutoff

prop_cutoff = 0.10

df = data.frame(mitochondria = mito_total, total=total)
df$prop_mito = df$mitochondria/df$total

g1 = ggplot(df, aes(x=total, y=mitochondria)) +
  geom_pointdensity() + scale_color_viridis() + 
  geom_hline(yintercept = total_cutoff) + 
  geom_abline(intercept = 0, slope = prop_cutoff)

g2 = ggplot(df, aes(x=prop_mito)) +
  geom_histogram(color="darkblue", fill="lightblue", bins=50) + 
  geom_vline(xintercept = prop_cutoff)

p1 = ggarrange(g1, g2, ncol=1, heights=c(4,3))
ggsave(p1, file=sprintf("figures/check_mito_%s.png", cell_type), 
       width=5, height=7)

table(mito_total > total_cutoff)
table(mito_total > total_cutoff, df$prop_mito > prop_cutoff)

w2rm = which(mito_total > total_cutoff | df$prop_mito > prop_cutoff)

gex_count = gex_count[-w2rm,]
cell_info = cell_info[-w2rm,]

dim(gex_count)

dim(cell_info)
cell_info[1:2,]

gex_count = as(gex_count, "sparseMatrix")

writeMM(gex_count, file=sprintf("gex_%s_BL.mtx", cell_type))
system(sprintf("gzip -f gex_%s_BL.mtx", cell_type))

fwrite(data.frame(gene = colnames(gex_count)), col.names = FALSE,
       file=sprintf("gex_%s_BL_genes.txt", cell_type))

fwrite(cell_info, file=sprintf("cell_info_%s_BL.csv", cell_type))

gc()

sessionInfo()
q(save="no")
