
using NearestNeighbors
using GeomDiagnostics.Samplers
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

covariance_field = Samplers.construct_ngpcacf_from_data(centers, Samplers.UseBallTree())

@testset "Can instantiate NGPCA Covariance field given a matrix of centroid columns" begin
    @test covariance_field isa Samplers.AbstractCovarianceField
end

@testset "Covariance field provides nearest neighbor" begin
    query1 = Vector([1.1, 2.2])
    query2 = Vector([3.1, 4.2])
    
    idx1, dist1 = Samplers.get_knn(query1, covariance_field, 1)
    idx2, dist2 = Samplers.get_knn(query2, covariance_field, 1)

    # Julia Matrix are 1 indexed
    @test idx1 == [1]
    @test idx2 == [2]

end