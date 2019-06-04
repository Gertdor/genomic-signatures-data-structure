import argparse
import pickle
from collections import defaultdict

from clustering_genomic_signatures.dbtools.get_signature_metadata import (
    get_metadata_for,
)
from scipy import stats
from ete3 import NCBITaxa

rank_dict = {
        'species':'genus',
        'superkingdom': None,
        'class': None,
        'subspecies': 'genus',
        'phylum': None,
        'no rank': 'unknown',
        'genus': 'genus',
        'subtribe': 'family',
        'tribe': 'family',
        'family': 'family',
        'subfamily': 'family',
        'subclass': None,
        'superorder': None,
        'order': None,
        'superclass': None,
        'species group': 'genus',
        'varietas': 'genus',
        'parvorder': None,
        'subgenus': 'genus',
        'infraorder': None,
        'species subgroup': 'genus',
        'infraclass': None,
        'kingdom': None,
        'superfamily': None,
        'forma': 'genus',
        'subkingdom': None,
        'suborder': None,
        'cohort': None,
        'subphylum': None,
        'section': None
        }

def calculate_classification_accuracy(results, meta_data):
    
    ncbi = NCBITaxa()
    total_accuracy = []
    ranks = defaultdict(lambda:[])
    classification_accuracy = defaultdict(list)
    failed_to_classify = 0
    
    for key, values in results.items():
        true_labels = meta_data[key]
        for taxid in values:
            if taxid != '0' and taxid != '1':
                rank, name = get_classification_rank_and_taxid(ncbi, taxid)
                if rank is not None:
                    classification_accuracy[rank].append(name == true_labels[rank])
                else:
                    failed_to_classify+=1
            else:
                failed_to_classify+=1
        
    print(sum(classification_accuracy["genus"])/len(classification_accuracy["genus"]))
    print(sum(classification_accuracy["family"])/len(classification_accuracy["family"]))
    print(failed_to_classify)

def get_classification_rank_and_taxid(ncbi, taxid):
    tmp_dict = ncbi.get_rank([taxid])
    if not tmp_dict:
        return (None, None)
    _, rank = tmp_dict.popitem() # returns list of values and takes list

    wanted_rank = rank_dict[rank]
    
    if(wanted_rank is "unknown"):
        wanted_rank = get_classification_level(ncbi, taxid)

    if(wanted_rank is not None):
        _, name = get_name_of_rank_in_lineage(ncbi, wanted_rank, taxid)
        return (wanted_rank, name)
    else:
        return (None, None)
        
def get_classification_level(ncbi, taxid):
    lineage = ncbi.get_lineage(taxid)
    for curr_id in reversed(lineage):
        _, rank = ncbi.get_rank([curr_id]).popitem()
        if rank_dict[rank] is not "unknown":
            return rank_dict[rank]
    
    return None

def get_name_of_rank_in_lineage(ncbi,rank,taxid):
    lineage = ncbi.get_lineage(taxid)
    for curr_id in reversed(lineage):
        _, new_rank = ncbi.get_rank([curr_id]).popitem() 
        if new_rank == rank:
            return ncbi.get_taxid_translator([curr_id]).popitem()
    if(rank=="genus"):
        return get_name_of_rank_in_lineage(ncbi,"family",taxid) #Very ugly fix. Some species lack genus
    
    return None,"notInDB"

parser = argparse.ArgumentParser()

parser.add_argument(
    "--classification_results", help="file containing hash map generated with reduce_output.py"
)

parser.add_argument(
    "--db_config", help="path to db_config for neo4j database"
)

args = parser.parse_args()

with open(args.classification_results,"rb") as f:
    results = pickle.load(f)

with open("../settings/query_aids.txt","r") as f:
    names = [n.strip() for n in f.readlines()]

meta_data = get_metadata_for(names,args.db_config)

#with open("meta_data.pickle","wb") as f:
#    pickle.dump(metadata,f)

#with open("meta_data.pickle","rb") as f:
#    metadata = pickle.load(f)

calculate_classification_accuracy(results, meta_data)
