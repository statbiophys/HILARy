library(scoper)
library(readr)
library(progress)

files <- list.files(path = "data/subsampled_benchmark/")

for(file in files) {
  db <- read_csv(gzfile(paste("data/subsampled_benchmark/",file,sep="")))
  colnames(db) <- c("sequence_alignment", "germline_alignment_d_mask", "v_call", "j_call", "junction_len", "junction", "family")
  time_taken <- system.time({dbsc <- spectralClones(db)})
  print(time_taken)
  write.csv(dbsc@db, paste("data/scoper_benchmark/", file, sep=""))
}
				