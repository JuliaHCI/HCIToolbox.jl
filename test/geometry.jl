

@testset "AnnulusView" begin
    data = ones(10, 501, 501)

    av = AnnulusView(data; inner=10, outer=30)
    
    @test size(av) == size(data)
    @test 10 * π * (av.rmax^2 - av.rmin^2) ≈ sum(av) rtol=1/sqrt(sum(av))

    X = flatten(av)
    @test all(size(X) .== map(length, av.indices))
    @test all(isone, X)
    sub_cube = inverse(av, -ones(size(X)))
    @test sub_cube ≈ copyto!(copy(av), -ones(size(X)))

    @test - 10 * π * (av.rmax^2 - av.rmin^2) ≈ sum(sub_cube) rtol=1/sqrt(count(!iszero, sub_cube))

    data = randn(rng, 10, 501, 501)

    @test AnnulusView(data; inner=0, outer=Inf) ≈ data

    nanview = AnnulusView(data; fill=NaN)
    @test_throws BoundsError nanview[-1, 2, 3]
    @test isnan(nanview[1, 1, 1])
end