@testset "klip" begin
    fake_cube = DataCube(ones(100, 100, 10), 50rand(10))

    reduced_5 = reduce(PCA, fake_cube, maxoutdim = 5)
    reduced_10 = reduce(PCA, fake_cube, maxoutdim = 10)

    @test size(reduced_5) == size(reduced_10) == (100, 100)
    @test reduced_5 == reduced_10 == zeros(100, 100)

end
