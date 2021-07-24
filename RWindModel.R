args = commandArgs(trailingOnly=TRUE)
library(stormwindmodel)

loc_data <- read.csv(args[1])
kwinds <- get_grid_winds(hurr_track = eval(parse(text=args[2])) , grid_df = loc_data)
write.csv(kwinds, args[3])