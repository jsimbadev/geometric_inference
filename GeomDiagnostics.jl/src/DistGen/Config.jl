struct DistGenConfig
    Source::String
    Seed::Int
    PlotPoints::Bool
    SourceConfig::Dict{String, Any}
end

function _as_dict(obj)
    out = Dict{String, Any}()
    for (k, v) in obj
        out[String(k)] = v
    end
    out
end

function _get_real(cfg::Dict{String, Any}, key::String, default::Real)
    if !haskey(cfg, key)
        return float(default)
    end
    float(cfg[key])
end

function _get_int(cfg::Dict{String, Any}, key::String, default::Int)
    if !haskey(cfg, key)
        return default
    end
    Int(cfg[key])
end

function _get_vector_float(cfg::Dict{String, Any}, key::String, default::Vector{Float64})
    if !haskey(cfg, key)
        return default
    end
    [Float64(v) for v in cfg[key]]
end

function parse_distgen_config(raw_cfg::AbstractDict)
    cfg = _as_dict(raw_cfg)
    source = String(cfg["source"])
    seed = _get_int(cfg, "seed", 42)
    plot_points = get(cfg, "plot_points", false)
    source_cfg = haskey(cfg, "source_config") ? _as_dict(cfg["source_config"]) : Dict{String, Any}()
    DistGenConfig(source, seed, Bool(plot_points), source_cfg)
end

function read_distgen_config(config_path::AbstractString)
    parse_distgen_config(Io.read_json(config_path))
end

function build_sample_source(cfg::DistGenConfig)
    source_name = cfg.Source
    source_cfg = cfg.SourceConfig

    if source_name == "normal"
        return get_sample_source("normal")
    end

    if source_name == "banana"
        a = _get_real(source_cfg, "a", 1.0)
        b = _get_real(source_cfg, "b", 1.0)
        base = Distributions.MvNormal([0, 0], reshape([1.0, 0.9, 0.9, 1.0], 2, 2))
        return BananaSampleSource(a, b, base)
    end

    if source_name == "pdrwm_normal2d"
        proposal_variance = _get_real(source_cfg, "proposal_variance", 0.5)
        burnin = _get_int(source_cfg, "burnin", 200)
        thinning = _get_int(source_cfg, "thinning", 1)
        initial_state = _get_vector_float(source_cfg, "initial_state", [0.0, 0.0])

        sampler = make_pdrwm_sampler_2d(; proposal_variance)
        target = MvNormalLogDensity(Distributions.MvNormal([0.0, 0.0], Matrix{Float64}(I, 2, 2)))
        model = AbstractMCMC.LogDensityModel(target)
        return MCMCSampleSource(model, sampler, initial_state, burnin, thinning)
    end

    if source_name == "pdrwm_banana2d"
        proposal_variance = _get_real(source_cfg, "proposal_variance", 0.5)
        burnin = _get_int(source_cfg, "burnin", 200)
        thinning = _get_int(source_cfg, "thinning", 1)
        initial_state = _get_vector_float(source_cfg, "initial_state", [0.0, 0.0])
        a = _get_real(source_cfg, "a", 1.0)
        b = _get_real(source_cfg, "b", 1.0)

        sampler = make_pdrwm_sampler_2d(; proposal_variance)
        base = Distributions.MvNormal([0.0, 0.0], reshape([1.0, 0.9, 0.9, 1.0], 2, 2))
        target = BananaLogDensity(a, b, base)
        model = AbstractMCMC.LogDensityModel(target)
        return MCMCSampleSource(model, sampler, initial_state, burnin, thinning)
    end

    if source_name == "rwm_banana2d"
        proposal_variance = _get_real(source_cfg, "proposal_variance", 0.5)
        burnin = _get_int(source_cfg, "burnin", 200)
        thinning = _get_int(source_cfg, "thinning", 1)
        initial_state = _get_vector_float(source_cfg, "initial_state", [0.0, 0.0])
        a = _get_real(source_cfg, "a", 1.0)
        b = _get_real(source_cfg, "b", 1.0)

        dimension = _get_int(source_cfg, "dim", 2)
        sampler = make_rwm_sampler_2d(; proposal_variance, dimension)
        base = Distributions.MvNormal([0.0, 0.0], reshape([1.0, 0.9, 0.9, 1.0], 2, 2))
        target = BananaLogDensity(a, b, base)
        model = AbstractMCMC.LogDensityModel(target)
        return MCMCSampleSource(model, sampler, initial_state, burnin, thinning)
    end

    get_sample_source(source_name)
end
