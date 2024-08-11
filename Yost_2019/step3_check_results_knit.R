
Sys.setenv(RSTUDIO_PANDOC="/Applications/RStudio.app/Contents/MacOS/pandoc")

render_report = function(graph_weight) {

  rmarkdown::render(
    "step3_check_results.Rmd", 
    params = list(graph_weight=graph_weight),
    output_file = paste0("step3_check_results_", graph_weight, ".html")
  )
}

weights = c("0.0", "0.25", "0.5", "1.0", "2.0", "5.0", "10.0")

for(graph_weight in weights){
  render_report(graph_weight)
}

sessionInfo()

q(save = "no")
