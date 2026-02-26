module Samplers

using ..NGPCAJson
using NearestNeighbors, HNSW
 # TODO keep this file backend-agnostic via wrapper constructors (exact vs approximate)

abstract type Atlas end

abstract type CovarianceField <: Atlas end

abstract type AbstractNearestNeighbourLookup end

abstract type ExactNN <: AbstractNearestNeighbourLookup end

abstract type ApproximateNN <: AbstractNearestNeighbourLookup end


struct HNSWNN <: ApproximateNN
    # TODO include metric/config fields if we need reproducible ANN settings
    index::HierarchicalNSW
end

struct BallTreeNN <: ExactNN
    # TODO keep point layout consistent (NearestNeighbors expects points in columns)
    index::BallTree
end


struct NGPCACF{NN<:AbstractNearestNeighbourLookup} <: CovarianceField
    nn::NN
    # TODO keep units in this struct so neighbour ids can map to covariance payload
    # units::Vector{NGPCAUnit}
end

function construct_ngpcacf_from_disk(ngpca_file::String)
    units = NGPCAJson.read_ngpca_units(ngpca_file)
    # TODO map units -> center matrix `data` before building index
    # TODO start with exact baseline (BallTreeNN), then add HNSWNN path
    # TODO return NGPCACF(nn=..., units=...) once payload wiring is done
    # Euclidean is default but want to be explicit
    nn_index = HNSWNN(HierarchicalNSW(data; metric=Euclidean()))

    
end

struct LocalCovarianceRWM{CF<: Atlas}
    # TODO add proposal configuration here (step scale, warmup/adaptation flags)
    field::CF
end




end
