module Samplers

using ..NGPCAJson
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

# TODO naming here 'data' is not the best... think over eventually
# Do I want convernience methods to go from full fat NGPCA object in memory?
function construct_ngpcacf_from_data(data::AbstractMatrix{<:Real}, backend::NNBackend)
    _construct_cf(Matrix{Float64}(data), backend)
end

function _construct_cf(data::Matrix{Float64}, backend::UseBallTree)
    nn = BallTreeNN(BallTree(data; parallel=backend.parallel))
    NGPCACF{typeof(nn)}(nn)
end


function _construct_cf(data::Matrix{Float64}, backend::UseHNSW)
    #  https://github.com/JuliaNeighbors/HNSW.jl
    # TODO Making a copy here, probably dont want to do this eventually 
    points = [Vector(data[:, i]) for i in axes(data, 2)]
    hnsw_index = HierarchicalNSW(points; metric=backend.metric)
    add_to_graph!(hnsw_index)
    nn = HNSWNN(hnsw_index)
    NGPCACF{typeof(nn)}(nn)
end


# https://github.com/JuliaIO/JSON.jl/blob/f4fbb5a429a21b422c88883981c34e29d22b887e/src/parse.jl#L33
function construct_ngpcacf_from_disk(ngpca_file::AbstractString, backend::NNBackend)
    units = NGPCAJson.read_ngpca_units(ngpca_file)
    array_based_units = NGPCAJson.array_of_units_to_array_based(units)
    construct_ngpcacf_from_data(array_based_units.centers, backend)
end

struct LocalCovarianceRWM{CF<:AbstractCovarianceField}
    field::CF
end


function get_knn(q::AbstractVector, cf::AbstractCovarianceField, k::Int=1)
    get_knn(q, cf.nn, k)
end

function get_knn(q::AbstractVector, nn::HNSWNN, k::Int=1)
    knn_search(nn.index, q, k)
end

function get_knn(q::AbstractVector, nn::BallTreeNN, k::Int=1)
    knn(nn.index, q, k)
end

end
