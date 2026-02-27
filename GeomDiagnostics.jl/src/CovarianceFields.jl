module CovarianceFields

using ..NGPCAJson
using NearestNeighbors, HNSW

include("CovarianceFields/Types.jl")
include("CovarianceFields/Construction.jl")
include("CovarianceFields/Query.jl")

using .Types: Atlas,
    AbstractCovarianceField,
    AbstractNearestNeighbourLookup,
    ExactNN,
    ApproximateNN,
    HNSWNN,
    BallTreeNN,
    NNBackend,
    UseHNSW,
    UseBallTree,
    NGPCACF,
    LocalCovarianceRWM

using .Construction: construct_ngpcacf_from_data,
    construct_ngpcacf_from_disk

using .Query: get_knn

end
