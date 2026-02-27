module Query

using NearestNeighbors, HNSW
using ..Types

function get_knn(q::AbstractVector, cf::Types.AbstractCovarianceField, k::Int=1)
    get_knn(q, cf.nn, k)
end

function get_knn(q::AbstractVector, nn::Types.HNSWNN, k::Int=1)
    knn_search(nn.index, q, k)
end

function get_knn(q::AbstractVector, nn::Types.BallTreeNN, k::Int=1)
    knn(nn.index, q, k)
end

end
