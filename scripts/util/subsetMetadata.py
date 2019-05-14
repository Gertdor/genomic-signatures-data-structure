def subset_metadata(dictionary):
    values_to_keep = ["order","family","genus","species"]
    return {k:dictionary[k] for k in values_to_keep}

def subset_all_metadata(dictionary):
    return {k:subset_metadata(v) for k,v in dictionary.items()}
