module Io
# Some looking into io best practice
# https://discourse.julialang.org/t/what-is-the-io-interface/107350


using CSV, Tables

function write_samples(samples::AbstractMatrix, file_path::String)
    CSV.write(file_path, Tables.table(samples), writeheader=false)
end

end