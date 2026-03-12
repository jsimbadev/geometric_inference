
using AbstractMCMC
using LogDensityProblems
using ..NGPCAJson
using ..CovarianceFields
using ..CovarianceFields.Types: AbstractCovarianceField
using ..CovarianceFields.Query: get_knn
using Distributions, Random, LinearAlgebra

abstract type AbstractPositionDependentRWMSampler <: AbstractMCMC.AbstractSampler end

struct HardPositionDependentRWMSampler{CF<:AbstractCovarianceField} <: AbstractPositionDependentRWMSampler
    CovF::CF
    Metadata::NGPCAJson.NGPCAUnitDO
end


function AbstractMCMC.step(rng::AbstractRNG, model::AbstractMCMC.LogDensityModel, sampler::AbstractPositionDependentRWMSampler; kwargs...)
    initial_state = get(kwargs, :initial_state, zeros(dimension(sampler)))
    AbstractMCMC.step(rng, model, sampler, initial_state; kwargs...)
end

function AbstractMCMC.step(rng::AbstractRNG, model::AbstractMCMC.LogDensityModel, sampler::AbstractPositionDependentRWMSampler, state; kwargs...)
    proposal = propose(rng, state, sampler)
    proposed_state = proposal.state
    forward_logproposal = proposal.logproposal

    current_logdensity = target_logdensity(model, state)
    proposed_logdensity = target_logdensity(model, proposed_state)
    reverse_logproposal = proposal_logdensity(proposed_state, state, sampler)

    log_acceptance = proposed_logdensity + reverse_logproposal - current_logdensity - forward_logproposal
    if log(rand(rng)) < min(0.0, log_acceptance)
        return proposed_state
    end

    state
end

function AbstractMCMC.step(rng::AbstractRNG, model::AbstractMCMC.AbstractModel, sampler::AbstractPositionDependentRWMSampler; kwargs...)
    error("$(typeof(sampler)) expects model::AbstractMCMC.LogDensityModel. Wrap your target with AbstractMCMC.LogDensityModel.")
end

function AbstractMCMC.step(rng::AbstractRNG, model::AbstractMCMC.AbstractModel, sampler::AbstractPositionDependentRWMSampler, state; kwargs...)
    error("$(typeof(sampler)) expects model::AbstractMCMC.LogDensityModel. Wrap your target with AbstractMCMC.LogDensityModel.")
end

function covfield(sampler::HardPositionDependentRWMSampler)
    sampler.CovF
end

function metadata(sampler::HardPositionDependentRWMSampler)
    # Metadata should be some integer position indexable
    # sturcture that lines up with the position of the 
    # centroids inside the nearest neighbour lookup
    sampler.Metadata
end

function dimension(sampler::HardPositionDependentRWMSampler)
    size(metadata(sampler).centers, 1)
end

function dimension(s::AbstractPositionDependentRWMSampler)
    error("Not Implemented for $(typeof(s))")
end

function propose(rng::AbstractRNG, state::AbstractVector, sampler::HardPositionDependentRWMSampler)
    covariance = local_covariance(state, sampler)
    proposal_dist = MvNormal(state, covariance)
    proposed_state = rand(rng, proposal_dist)
    (state=proposed_state, covariance=covariance, logproposal=logpdf(proposal_dist, proposed_state))
end

function propose(rng::AbstractRNG, state::AbstractVector, sampler::AbstractPositionDependentRWMSampler)
    error("Not implemented for $(typeof(sampler))")
end

function proposal_distribution(state::AbstractVector, sampler::HardPositionDependentRWMSampler)
    MvNormal(state, local_covariance(state, sampler))
end

function proposal_logdensity(from::AbstractVector, to::AbstractVector, sampler::HardPositionDependentRWMSampler)
    logpdf(proposal_distribution(from, sampler), to)
end

function local_covariance(state::AbstractVector, sampler::HardPositionDependentRWMSampler)
    idx, _ = get_knn(state, covfield(sampler), 1)
    construct_covariance(idx[begin], metadata(sampler))
end

function target_logdensity(model::AbstractMCMC.LogDensityModel, state::AbstractVector)
    LogDensityProblems.logdensity(model.logdensity, state)
end

function construct_covariance(i::Int, indexableMetadata::NGPCAJson.NGPCAUnitDO)
    eigvals, eigvecs = indexableMetadata.eigenvalues[:, i], indexableMetadata.weights[i]
    stable_eigvals = max.(eigvals, zero(eltype(eigvals)))
    covariance = eigvecs * Diagonal(stable_eigvals) * transpose(eigvecs)
    jitter = max(indexableMetadata.epsilons[i], eps(eltype(covariance)))
    Matrix(Symmetric(covariance + jitter * I))
end
