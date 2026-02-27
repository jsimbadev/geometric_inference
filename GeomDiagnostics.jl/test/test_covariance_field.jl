
using NearestNeighbors
using JSON
using GeomDiagnostics.CovarianceFields
import GeomDiagnostics.NGPCAJson: NGPCAUnitDO

m=Vector([1,2])
centers=reshape([1.0, 2.0, 3.0, 4.0], 2,2)
weights=Vector([reshape([1.0, 2.0, 3.0, 4.0], 2,2), reshape([1.0, 2.0, 3.0, 4.0], 2,2)])
eigenvalues=reshape([1.0, 2.0, 3.0, 4.0], 2,2)
sigma_sqrs=Vector([2,1])
activities=Vector([3,4])
alphas=Vector([5,6])
epsilons=Vector([7,8])

test_covriance_data = NGPCAUnitDO(
    m,
    centers,
    weights,
    eigenvalues,
    sigma_sqrs,
    activities,
    alphas,
    epsilons
)


@testset "Can instantiate NGPCA Covariance field given a matrix of centroid columns" begin
    covariance_field = CovarianceFields.construct_ngpcacf_from_data(centers, CovarianceFields.UseBallTree())
    @test covariance_field isa CovarianceFields.AbstractCovarianceField
end

@testset "Can instantiate NGPCA Covariance field with HNSW backend" begin
    covariance_field_hnsw = CovarianceFields.construct_ngpcacf_from_data(centers, CovarianceFields.UseHNSW(Euclidean()))
    @test covariance_field_hnsw isa CovarianceFields.AbstractCovarianceField

    idx, dist = CovarianceFields.get_knn(Vector([1.1, 2.2]), covariance_field_hnsw, 1)
    @test idx == [1]
end

@testset "Can instantiate NGPCA Covariance field from disk JSON" begin
    tmp_json = tempname() * ".json"
    units_obj = Dict(
        "units" => [
            Dict(
                "m" => 2,
                "center" => [1.0, 2.0],
                "weight" => [[1.0, 0.0], [0.0, 1.0]],
                "eigenvalue" => [1.0, 1.0],
                "sigma_sqr" => 1.0,
                "activity" => 1.0,
                "alpha" => 0.1,
                "epsilon" => 0.1
            ),
            Dict(
                "m" => 2,
                "center" => [3.0, 4.0],
                "weight" => [[1.0, 0.0], [0.0, 1.0]],
                "eigenvalue" => [1.0, 1.0],
                "sigma_sqr" => 1.0,
                "activity" => 1.0,
                "alpha" => 0.1,
                "epsilon" => 0.1
            )
        ]
    )
    write(tmp_json, JSON.json(units_obj))

    covariance_field_disk = CovarianceFields.construct_ngpcacf_from_disk(tmp_json, CovarianceFields.UseBallTree())
    @test covariance_field_disk isa CovarianceFields.AbstractCovarianceField
    idx, dist = CovarianceFields.get_knn(Vector([3.1, 4.2]), covariance_field_disk, 1)
    @test idx == [2]

    rm(tmp_json; force=true)
end

@testset "Covariance field provides nearest neighbor" begin
    query1 = Vector([1.1, 2.2])
    query2 = Vector([3.1, 4.2])

    covariance_field = CovarianceFields.construct_ngpcacf_from_data(centers, CovarianceFields.UseBallTree())
    
    idx1, dist1 = CovarianceFields.get_knn(query1, covariance_field, 1)
    idx2, dist2 = CovarianceFields.get_knn(query2, covariance_field, 1)

    # Julia Matrix are 1 indexed
    @test idx1 == [1]
    @test idx2 == [2]

end
