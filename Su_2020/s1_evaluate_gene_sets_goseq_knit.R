
Sys.setenv(RSTUDIO_PANDOC="/Applications/RStudio.app/Contents/MacOS/pandoc")

render_report = function(cell_type_name, graph_weight) {
  file_tag = paste0(cell_type_name, "_BL_", graph_weight)

  rmarkdown::render(
    "s1_evaluate_gene_sets_goseq.Rmd", 
    params = list(cell_type_name = cell_type_name, graph_weight=graph_weight),
    output_file = paste0("s1_evaluate_gene_sets_goseq_", file_tag, ".html")
  )
}

cell_types = c("cd4", "cd8")
weights = list()
weights[["cd4"]] = c("0.0", "0.25", "0.5", "1.0", "2.0", "5.0", "10.0")
weights[["cd8"]] = c("0.0", "0.25", "0.5", "1.0", "2.0", "5.0", "10.0")

for(cell_type_name in cell_types){
  for(graph_weight in weights[[cell_type_name]]){
    render_report(cell_type_name, graph_weight)
  }
}


sessionInfo()

q(save = "no")
