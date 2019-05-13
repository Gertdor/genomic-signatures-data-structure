#! /bin/bash
docker build --rm \
    -t gs-data-structure \
    .

docker run --rm \
    --net=host \
    --env="DISPLAY" \
    -v "$HOME"/.Xauthority:/home/genomicsignatures/.Xauthority:rw \
    -v "$PWD/scripts":/home/genomicsignatures/genomic-signatures-data-structures/scripts \
    -v "$PWD/gs-data-structures":/home/genomicsignatures/genomic-signatures-data-structures/gs-data-structures \
    -v "$PWD/tests":/home/genomicsignatures/genomic-signatures-data-structures/tests \
    -it gs-data-structure \
    /bin/bash -c $'cd scripts && /bin/bash'
