from dataStructures.VLMCElement import VPTreeVLMC
from util.subsetMetadata import subset_all_metadata

from clustering_genomic_signatures.util.parse_signatures import (
    parse_signatures,
    add_parse_signature_args,
)
from clustering_genomic_signatures.util.parse_distance import (
    add_distance_arguments,
    parse_distance_method,
)
from clustering_genomic_signatures.dbtools.get_signature_metadata import (
    get_metadata_for,
)

def generate_vlmc_elements(args):
    vlmcs = parse_signatures(args, args.db_config)
    names = [vlmc.name for vlmc in vlmcs]
    meta_data = subset_all_metadata(get_metadata_for(names, args.db_config))
    
    distance_function = parse_distance_method(args)
    tmp = args.distance_function
    args.distance_function = "gc-content"
    fast_dist = parse_distance_method(args)
    args.distance_function = tmp
    
    elements = [
        VPTreeVLMC(vlmc, distance_function, fast_dist = fast_dist, taxonomic_data = meta_data[name])
        for name,vlmc in zip(names,vlmcs)
    ]
    return elements

def add_generate_vlmc_args(parser):
    add_parse_signature_args(parser)
    add_distance_arguments(parser)
