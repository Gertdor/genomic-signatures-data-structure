import os
import re
import sys
import gzip

def find_latest_assembly(root_path, output_dir):
    if not os.path.isdir(root_path):
        return
    for path in os.scandir(root_path):
        if "latest_assembly_versions" == path.name:
            find_report(path,output_dir)
        if path.is_dir():
            if "all_assembly_versions" == path.name:
                continue
            find_latest_assembly(path, output_dir)

def find_report(root_path, output_dir):
    if not os.path.isdir(root_path):
        return False
    for path in os.scandir(root_path):
        if re.search("report\.txt",path.name):
            if is_large_enough(path.path):
                unpack_and_save(root_path, output_dir)
                return True
        if path.is_dir():
            if find_report(path, output_dir):
                return True
    return False

def is_large_enough(report_location):
    with open(report_location,"r") as f:
        for line in f:
            if re.search("Assembly level: Chromosome|Assembly level: Complete Genome",line):
                return True
        return False

def unpack_and_save(root_path,output_dir):
    for path in os.scandir(root_path):
        if re.search("genomic\.fna\.gz",path.name):
            if not re.search("_cds_|_rna_", path.name):
                organism_name = path.path.split('/')[-2]
                with gzip.open(path.path,'rb') as gz_f:
                    with open(output_dir+"/"+organism_name + ".fna",'wb') as f:
                        content = gz_f.read()
                        f.write(content)


def list_all_relevant_paths(root_path, output_dir):
    find_latest_assembly(root_path,output_dir)



list_all_relevant_paths(sys.argv[1], sys.argv[2])
