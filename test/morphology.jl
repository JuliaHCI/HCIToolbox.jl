@testset "flatten/expand" begin

    @testset "fllatten" begin
        @test flatten(ones(3, 4, 4)) == ones(3, 16)

        @test_throws MethodError flatten(ones(3, 4))
    end

    @testset "expand" begin
        @test expand(ones(3, 16)) == ones(3, 4, 4)

        @test_throws ErrorException expand(ones(3, 15))
    end

    # Simple regression
    X = rand(10, 512, 512)
    @test expand(flatten(X)) == X
end

@testset "rotation" begin
    @testset "rotate" begin
        X = rand(10, 512, 512)

        # test no-op
        @test rotate(X, zeros(10)) === X
    end
    @testset "derotate" begin
        X = rand(10, 512, 512)
        
        # test no-op
        @test derotate(X, zeros(10)) === X
    end

    # Simple regression (but be careful to only test some inner part that won't be clipped)
    X = rand(10, 512, 512)
    θ = 90 .* rand(10)
    @test filter(!isnan, derotate(X, θ)) ≈ filter(!isnan, rotate(X, -θ))
    @test filter(!isnan, rotate(X, θ)) ≈ filter(!isnan, derotate(X, -θ))

    targ = deepcopy(X)
    derotate!(X, θ)
    @test X !== targ
    X = rand(10, 512, 512)
    rotate!(X, θ)
    @test X !== targ

end

@testset "combining" begin
    X = rand(10, 512, 512)
    angles = sort!(90 .* rand(10))

    @test combine(X) == combine(X, zeros(10))
    @test combine(X) == combine(X; method = median)
    @test combine(X; method = mean) == mean(X, dims = 1)[1, :, :]

    @test filter(!isnan, combine(X, angles)) ≈ filter(!isnan, combine(derotate(X, angles)))
    @test any(combine(X, angles) .=== NaN)

end
