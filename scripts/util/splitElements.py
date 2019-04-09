import numpy as np


def split_elements(elems, args):

    if not args.no_randomize_elements:
        elements = elems.copy()
        np.random.shuffle(elements)
    else:
        elements = elems

    num_elem_in_tree = round(len(elements) * args.cutoff)
    if (
        args.number_of_searches == 0
        or args.number_of_searches + num_elem_in_tree > len(elements)
    ):
        args.number_of_searches = len(elements) - num_elem_in_tree

    tree_elements = elements[0:num_elem_in_tree]
    search_elements = elements[
        num_elem_in_tree : num_elem_in_tree + args.number_of_searches
    ]

    return (tree_elements, search_elements)
