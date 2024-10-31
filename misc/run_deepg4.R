library(optparse)
library(Biostrings)
library(rtracklayer)
library(DeepG4)

option_list = list(make_option(c("-i", "--input_file"), type="character", default=NULL,
			   		help="input sequences", metavar="character"),
				make_option(c("-o", "--output_file"), type="character", default=NULL,
			    	help="output file", metavar="character"),
		   		make_option(c("-k", "--scan_size"), type="integer", default=20,
			    	help="scan size for >201nt", metavar="scan_size"),
				make_option(c("-t", "--threshold"), type="double", default=0.5,
			    	help="threshold G4 score to keep", metavar="threshold")
)

opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

file_obj <- file(opt$input_file)
sequences <- readLines(file_obj); close(file_obj)
sequences <- DNAStringSet(sequences)

predictions <- DeepG4(X=sequences)
write(predictions, file=opt$output_file, sep='\n')