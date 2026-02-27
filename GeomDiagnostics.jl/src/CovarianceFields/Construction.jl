module Construction

using NearestNeighbors, HNSW
using ..Types
using ...NGPCAJson

# TODO naming here 'data' is not the best... think over eventually
# Do I want convernience methods to go from full fat NGPCA object in memory?
function construct_ngpcacf_from_data(data::AbstractMatrix{<:Real}, backend::Types.NNBackend)
    _construct_cf(Matrix{Float64}(data), backend)
end

function _construct_cf(data::Matrix{Float64}, backend::Types.UseBallTree)
    nn = Types.BallTreeNN(BallTree(data; parallel=backend.parallel))
    Types.NGPCACF{typeof(nn)}(nn)
end

function _construct_cf(data::Matrix{Float64}, backend::Types.UseHNSW)
    #  https://github.com/JuliaNeighbors/HNSW.jl
    # TODO Making a copy here, probably dont want to do this eventually
    points = [Vector(data[:, i]) for i in axes(data, 2)]
    hnsw_index = HierarchicalNSW(points; metric=backend.metric)
    add_to_graph!(hnsw_index)
    nn = Types.HNSWNN(hnsw_index)
    Types.NGPCACF{typeof(nn)}(nn)
end

# https://github.com/JuliaIO/JSON.jl/blob/f4fbb5a429a21b422c88883981c34e29d22b887e/src/parse.jl#L33
function construct_ngpcacf_from_disk(ngpca_file::AbstractString, backend::Types.NNBackend)
    units = NGPCAJson.read_ngpca_units(ngpca_file)
    array_based_units = NGPCAJson.array_of_units_to_array_based(units)
    construct_ngpcacf_from_data(array_based_units.centers, backend)
end

end
