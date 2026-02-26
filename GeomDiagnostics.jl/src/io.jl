module Io
# Some looking into io best practice
# https://discourse.julialang.org/t/what-is-the-io-interface/107350


using CSV, Tables, JSON

function write_samples(samples::AbstractMatrix, file_path::String)
    CSV.write(file_path, Tables.table(samples), writeheader=false)
end

function read_json(file::AbstractString)
    JSON.parsefile(file)
end

function read_json(file::AbstractString, ::Type{T}) where {T}
    JSON.parsefile(file, T)
end

function _read_json_field_path(obj, field_path::AbstractString, delim::AbstractString)
    if isempty(field_path)
        return obj
    end

    current = obj
    tokens = split(field_path, delim)
    for token in tokens
        if current isa AbstractDict
            current = current[String(token)]
        elseif current isa AbstractVector
            idx = parse(Int, token)
            current = current[idx]
        else
            error("Cannot descend into value at token '$token'")
        end
    end
    return current
end

function read_json(file::AbstractString, field_path::AbstractString, ::Type{T}; delim::AbstractString=".") where {T}
    obj = JSON.parsefile(file)
    sub_obj = _read_json_field_path(obj, field_path, delim)
    return JSON.parse(JSON.json(sub_obj), T)
end

end
