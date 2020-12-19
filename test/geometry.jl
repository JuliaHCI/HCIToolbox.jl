

@testset "AnnulusView" begin
    data = ones(10, 501, 501)

    av = AnnulusView(data; inner=10, outer=30)
    
    @test size(av) == size(data)
    @test 10 * π * (av.rmax^2 - av.rmin^2) ≈ sum(av) rtol=1/sqrt(sum(av))

    X = av()
    @test all(size(X) .== map(length, av.indices))
    @test all(isone, X)
    sub_cube = inverse(av, -ones(size(X)))
    @test sub_cube ≈ copyto!(copy(av), -ones(size(X)))

    X_view = av(true)
    @test X_view isa SubArray
    @test X_view ≈ X

    @test -10π * (av.rmax^2 - av.rmin^2) ≈ sum(sub_cube) rtol=1/sqrt(count(!iszero, sub_cube))

    data = randn(rng, 10, 501, 501)

    @test AnnulusView(data; inner=0, outer=Inf) ≈ data

    nanview = AnnulusView(data; fill=NaN)
    @test_throws BoundsError nanview[-1, 2, 3]
    @test isnan(nanview[1, 1, 1])
end


@testset "MultiAnnulusView" begin
    data = ones(10, 501, 501)
    fwhm = 5
    av = MultiAnnulusView(data, fwhm; inner=10, outer=30)

    @test size(av) == size(data)
    @test av.radii ≈ 12.5:5:27.5
    @test length(av.indices) == length(av.radii)
    @test av.width ≈ fwhm
    area = 10 * π * (30^2 - 10^2)
    @test area ≈ sum(av) rtol=1/sqrt(count(!iszero, av))

    X = av(1)
    @test all(size(X) .== map(length, av.indices[1]))
    @test all(isone, X)
    # only subs the first annulus
    @test_throws MethodError inverse(av, -X)
    @test_throws ErrorException copyto!(copy(av), -X)
    sub_cube = inverse(av, 1, -X)
    @test all(sub_cube[av.indices[1]...] .≈ -1)
    copy_cube = copyto!(copy(av), 1, -X)
    @test all(copy_cube[av.indices[1]...] .≈ -1)

    @test_throws DimensionMismatch inverse(av, 2, X)

    X_view = av(1, true)
    @test X_view isa SubArray
    @test X_view ≈ X

    av = MultiAnnulusView(data, fwhm; inner=10, outer=30)
    @test eachannulus(av) isa Base.Generator
    Xs = [-X for X in eachannulus(av)]
    sub_cube = inverse(av, Xs)
    @test sub_cube ≈ copyto!(copy(av), Xs)

    @test -area ≈ sum(sub_cube) rtol=1/sqrt(count(!iszero, sub_cube))

    data = randn(rng, 10, 501, 501)

    @test MultiAnnulusView(data, fwhm; inner=0, outer=Inf) ≈ data

    nanview = MultiAnnulusView(data, fwhm; fill=NaN)
    @test_throws BoundsError nanview[-1, 2, 3]
    @test isnan(nanview[1, 1, 1])
end

