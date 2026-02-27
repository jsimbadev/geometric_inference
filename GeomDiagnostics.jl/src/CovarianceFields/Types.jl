module Types

using NearestNeighbors, HNSW

abstract type Atlas end

abstract type AbstractCovarianceField <: Atlas end

abstract type AbstractNearestNeighbourLookup end

abstract type ExactNN <: AbstractNearestNeighbourLookup end

abstract type ApproximateNN <: AbstractNearestNeighbourLookup end

struct HNSWNN <: ApproximateNN
    index::HierarchicalNSW
end

struct BallTreeNN <: ExactNN
    index::BallTree
end

abstract type NNBackend end

# This configuration object is implemention bound
# So when underlying HNSW changes, this must too
struct UseHNSW <: NNBackend
    # TODO look into Distances.jl and get better typing
    metric::Euclidean
end

struct UseBallTree <: NNBackend
    parallel::Bool
    function UseBallTree()
        return new(false)
    end
end

struct NGPCACF{NN<:AbstractNearestNeighbourLookup} <: AbstractCovarianceField
    nn::NN
end

struct LocalCovarianceRWM{CF<:AbstractCovarianceField}
    field::CF
end

end
