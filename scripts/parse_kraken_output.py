import argparse
import pickle
from clustering_genomic_signatures.dbtools.get_signature_metadata import (
    get_metadata_for,
)
from scipy import stats

def calculate_classification_accuracy(results, metadata):
    
    total_accuracy = []
    for key, values in results.items():
        current_meta_data = metadata[key]
        species = current_meta_data["species"]
        print(species)
        genus = current_meta_data["genus"]
        family = current_meta_data["family"]
        subfamily = current_meta_data["subfamily"]
        order = current_meta_data["order"]

        species_accuracy = sum(d == species for d in values)/len(values)
        genus_accuracy = sum(d == genus for d in values)/len(values)
        family_accuracy = sum(d == family for d in values)/len(values)
        subfamily_accuracy = sum(d == subfamily for d in values)/len(values)
        order_accuracy = sum(d == order for d in values)/len(values)
        total_accuracy.append((species_accuracy, genus_accuracy,subfamily_accuracy, family_accuracy, order_accuracy))
    
    print(stats.describe(total_accuracy))
    
parser = argparse.ArgumentParser()

parser.add_argument(
    "--classification_results", help="file containing hash map generated with reduce_output.py"
)
parser.add_argument(
    "--kraken_db", help="root folder of the kraken db"
)
parser.add_argument(
    "--db_config", help="path to db_config for neo4j database"
)

args = parser.parse_args()

with open(args.classification_results,"rb") as f:
    results = pickle.load(f)

with open("query_aids.txt","r") as f:
    names = [n.strip() for n in f.readlines()]

metadata = get_metadata_for(names,args.db_config)

#with open("meta_data.pickle","wb") as f:
#    pickle.dump(metadata,f)

#with open("meta_data.pickle","rb") as f:
#    metadata = pickle.load(f)

calculate_classification_accuracy(results, metadata)
