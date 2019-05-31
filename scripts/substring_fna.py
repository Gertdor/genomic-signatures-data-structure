import argparse
import os

def splitfile(filename, args):
    file_root = filename.split("/")[-1]
    file_root = file_root.split(".")
    file_root = file_root[:-1]
    file_root = ".".join(file_root)
    try:
        with open(filename,"r") as f:
            complete_genome = f.readlines()
        
        number_of_samples = round(len(complete_genome)/args.read_length)
        if args.max_number_of_samples is not None:
            if(number_of_samples > args.max_number_of_samples):
                number_of_samples = args.max_number_of_samples
        
        #skip first line
        reads = [complete_genome[i*args.read_length:(i+1)*args.read_length] 
                    for i in range(1,number_of_samples)
                ]
        #dir_name = args.output_dir + "/" + file_root
        #if not os.path.exists(dir_name):
        #    os.mkdir(dir_name)
        with open(args.output_dir + '/' + file_root + ".fa","w") as f:
            for i,read in enumerate(reads):
                f.write(">"+file_root+" seq " + str(i) + "\n")
                f.write(''.join(read))
    except IOError as e:
        print(e)

parser = argparse.ArgumentParser(description = "arguments for fna string split")

parser.add_argument(
    "--read_length",
    type=int,
    default=2,
    help="number of rows to use for each read",
)
parser.add_argument(
    "--input_folder",
    help="folder containing fna files to split",
)
parser.add_argument(
    "--subset_text_file",
    default=None,
    help="file containing a list of species to classify. An fna file should be present" + 
            "with the same name in input_folder",
)
parser.add_argument(
    "--max_number_of_samples",
    type=int,
    default=10000,
    help="the number of samples to draw from each file. Default is all",
)
parser.add_argument(
    "--output_dir",
    default=None,
    help="name of root folder where the reads will be saved",
)
args = parser.parse_args()

if(args.subset_text_file):
    with open(args.subset_text_file,"r") as f:
        all_organisms = f.readlines()
        all_files = [args.input_folder + organism.strip() + ".fa" for organism in all_organisms]

[splitfile(file_name,args) for file_name in all_files]
