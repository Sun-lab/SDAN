
library(data.table)
library(stringr)

# ------------------------------------------------------------------------
# read in file information
# ------------------------------------------------------------------------

sdrf = fread("E-MTAB-9357.sdrf.txt")
dim(sdrf)
names(sdrf)
sdrf[1:2,1:30]

w_data_file = which(names(sdrf) == "Derived Array Data File")
w_data_file
sdrf[1:3,..w_data_file]

w_ftp = which(names(sdrf) == "Comment [Derived ArrayExpress FTP file]")
w_ftp

s0 = "ftp://ftp.ebi.ac.uk/pub/databases/microarray/"
s0 = paste0(s0, "data/experiment/MTAB/E-MTAB-9357/")

for(w1 in w_ftp){
  sdrf[[w1]] = gsub(s0, "", sdrf[[w1]])
}  
sdrf[1:3,..w_ftp]

# ------------------------------------------------------------------------
# keep file location information
# ------------------------------------------------------------------------

file_info = sdrf[,c(1:8,..w_data_file, ..w_ftp)]
dim(file_info)
file_info[1:2,]

names(file_info) = gsub("Characteristics[", "", names(file_info), fixed=TRUE)
names(file_info) = gsub("Factor Value[", "", names(file_info), fixed=TRUE)
names(file_info) = gsub("]", "", names(file_info))
names(file_info) = gsub(" ", "_", names(file_info))

sample_type = c("gex", "pro", "cd4_tcr", "cd8_tcr", "bcr")

names(file_info)[9:13]  = paste0(sample_type, "_file")
names(file_info)[14:18] = paste0(sample_type, "_dir")

dim(file_info)
file_info[1:2,]

table(file_info$gex_dir, file_info$disease)

table(file_info$developmental_stage)
table(file_info$sampling_time_point)
table(file_info$Material_Type)

gINC = grepl("INCOV", file_info$sampling_time_point)
table(gINC, file_info$disease)
fsub = file_info[which(gINC),]
fsub[1:2,]
table(fsub$disease)
table(fsub$gex_dir)
table(fsub$pro_dir)

ind_id = str_extract(file_info$sampling_time_point[gINC], 
                     "(\\S+)(?=-)")

ind_id[1:2]
table(ind_id == file_info$individual[gINC])

file_info$sampling_time_point[gINC] = 
  str_extract(file_info$sampling_time_point[gINC], "(?<=-)[:upper:]+")
table(file_info$sampling_time_point[gINC])
table(file_info$sampling_time_point)

fwrite(file_info, file="file_info.csv")

gc()

sessionInfo()
q(save="no")
