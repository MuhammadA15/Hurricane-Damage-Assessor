args = commandArgs(trailingOnly=TRUE)
library(stormwindmodel)
library(hurricaneexposuredata)

loc_data <- read.csv(args[1])
track <- args[2]
#track <- eval(parse(text=args[2]))

data("hurr_tracks")
track <- subset(hurr_tracks, storm_id==track)

kwinds <- get_grid_winds(hurr_track=track, grid_df=loc_data)
write.csv(kwinds, args[3])