module NGPCAJson

using ..Io

struct NGPCAUnit
    m::Int # PCADimensionality
    center::Vector{Float64}
    weight::Matrix{Float64}
    eigenvalue::Vector{Float64}
    sigma_sqr::Float64
    activity::Float64
    alpha::Float64
    epsilon::Float64
end

# DO - Data Oriented 
# Basically decouple the units into separate Sequential structures 
# data is assiciated by position in the sequence 
# For originally vector data, Matrix offers natural structure
struct NGPCAUnitDO
    # Vectors are best indexed in a matrix.
    m::Vector{Int}
    centers::Matrix{Float64}
    weights::Vector{Matrix{Float64}}
    eigenvalues::Matrix{Float64}
    sigma_sqrs::Vector{Float64}
    activities::Vector{Float64}
    alphas::Vector{Float64}
    epsilons::Vector{Float64}
end

struct NGPCAExport
    potentialFunction::String
    softhard::Float64
    learningRate::Float64
    rho_init::Float64
    rho_final::Float64
    mu::Float64
    rmax::Float64
    lambda::Float64
    numberUnits::Int
    PCADimensionality::Int
    iterations::Int
    rho::Float64
    units::Vector{NGPCAUnit}
end

function read_ngpca(file::AbstractString)
    Io.read_json(file, NGPCAExport)
end

function read_ngpca_units(file::AbstractString)
    Io.read_json(file, "units", Vector{NGPCAUnit})
end

function array_of_units_to_array_based(units::Vector{NGPCAUnit})
    n = length(units)
    if n == 0
        error("units must be non-empty")
    end

    d = length(units[1].center)
    m_dim = length(units[1].eigenvalue)

    m = Vector{Int}(undef, n)
    centers = Matrix{Float64}(undef, d, n)
    weights = Vector{Matrix{Float64}}(undef, n)
    eigenvalues = Matrix{Float64}(undef, m_dim, n)
    sigma_sqrs = Vector{Float64}(undef, n)
    activities = Vector{Float64}(undef, n)
    alphas = Vector{Float64}(undef, n)
    epsilons = Vector{Float64}(undef, n)

    for i in 1:n
        u = units[i]
        if length(u.center) != d
            error("all unit centers must have the same dimension")
        end
        if length(u.eigenvalue) != m_dim
            error("all unit eigenvalue vectors must have the same length")
        end

        m[i] = u.m
        centers[:, i] = u.center
        weights[i] = u.weight
        eigenvalues[:, i] = u.eigenvalue
        sigma_sqrs[i] = u.sigma_sqr
        activities[i] = u.activity
        alphas[i] = u.alpha
        epsilons[i] = u.epsilon
    end

    return NGPCAUnitDO(m, centers, weights, eigenvalues, sigma_sqrs, activities, alphas, epsilons)
end

end
