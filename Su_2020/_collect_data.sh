
R CMD BATCH --no-save --no-restore "--args cell_type='cd8'" _collect_data.R _collect_data_cd8.Rout &

R CMD BATCH --no-save --no-restore "--args cell_type='cd4'" _collect_data.R _collect_data_cd4.Rout &

