import os
import re


def explore_tree(root_path, stopping_name, output_file):
    if not os.path.isdir(root_path):
        if re.search(stopping_name[0], root_path.path):
            is_large_enough(root_path, output_file)
        return
    for path in os.scandir(root_path):
        if re.search(stopping_name[0], path.name):
            if len(stopping_name) == 1:
                is_large_enough(path.path, output_file)
            explore_tree(path, stopping_name[1:], output_file)
        if path.is_dir():
            explore_tree(path, stopping_name, output_file)


def is_large_enough(report_location, output_file):
    with open(report_location, "r") as f:
        for line in f:
            if re.search(
                "Assembly level: Chromosome|Assembly level: Complete Genome", line
            ):
                with open(output_file, "w") as f:
                    f.write(report_location)
                return
        return


def list_all_relevant_paths(root_path, output_file):
    explore_tree(root_path, ["latest_assembly_versions", "report\.txt"], output_file)


list_all_relevant_paths(
    "/home/basse/Documents/skola/masterThesis", "data/folder_locations.txt"
)
