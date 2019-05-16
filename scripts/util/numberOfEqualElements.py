def number_of_equal_elements(list1, list2):
    return sum([x == y for x, y in zip(list1, list2)])
