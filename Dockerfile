FROM python:3.7 as clustering-genomic-signatures

RUN apt-get update -y && apt-get install -y \
    graphviz \
    libblas-dev \
    liblapack-dev

RUN useradd -ms /bin/bash genomicsignatures -u 1000
USER genomicsignatures

# Install julia classifier
RUN mkdir -p /home/genomicsignatures/julia
WORKDIR /home/genomicsignatures/julia

RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.1/julia-1.1.0-linux-x86_64.tar.gz && \
    tar xf julia-1.1.0-linux-x86_64.tar.gz

ENV PATH "/home/genomicsignatures/julia/julia-1.1.0/bin:$PATH"

COPY --chown=genomicsignatures ./clustering-genomic-signatures-private/LazySuffixTrees.jl /home/genomicsignatures/julia/LazySuffixTrees.jl
COPY --chown=genomicsignatures ./clustering-genomic-signatures-private/PstClassifier.jl /home/genomicsignatures/julia/PstClassifier.jl

RUN julia -e 'using Pkg; Pkg.develop(PackageSpec(path="/home/genomicsignatures/julia/LazySuffixTrees.jl"))' && \
    julia -e 'using Pkg; Pkg.develop(PackageSpec(path="/home/genomicsignatures/julia/PstClassifier.jl"))' && \
    julia -e 'using Pkg; Pkg.add("PyCall")' &&  \
    julia --optimize=3 -e 'using PstClassifier'


RUN mkdir /home/genomicsignatures/genomic-signatures-data-structures
WORKDIR /home/genomicsignatures/genomic-signatures-data-structures

RUN wget ftp://ftp.genome.jp/pub/db/virushostdb/virushostdb.tsv

COPY --chown=genomicsignatures clustering-genomic-signatures-private clustering-genomic-signatures-private
COPY --chown=genomicsignatures gs-data-structures gs-data-structures

COPY --chown=genomicsignatures Makefile .
COPY --chown=genomicsignatures Pipfile .
COPY --chown=genomicsignatures Pipfile.lock .

ENV PATH "/home/genomicsignatures/.local/bin:$PATH"

RUN pip install pipenv --user
RUN pipenv install --dev --deploy --ignore-pipfile
