module NGPCAJson

using ..Io

struct NGPCAUnit
    m::Int
    center::Vector{Float64}
    weight::Matrix{Float64}
    eigenvalue::Vector{Float64}
    sigma_sqr::Float64
    activity::Float64
    alpha::Float64
    epsilon::Float64
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

end
