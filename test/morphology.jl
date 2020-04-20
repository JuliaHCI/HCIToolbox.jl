@testset "matrix/cube" begin

    @testset "matrix" begin
        @test matrix(ones(3, 4, 4)) == ones(3, 16)

        @test_throws MethodError matrix(ones(3, 4))
    end

    @testset "cube" begin
        @test cube(ones(3, 16)) == ones(3, 4, 4)

        @test_throws ErrorException cube(ones(3, 15))
    end

    # Simple regression
    X = rand(10, 512, 512)
    @test cube(matrix(X)) == X
end

@testset "derotate" begin
    X = rand(10, 512, 512)
    
    # test no-op
    @test derotate(X, zeros(10)) === X

    # Simple regression (but be careful to only test some inner part that won't be clipped)
    X = rand(10, 512, 512)
    θ = 90 .* rand(10)
    B = derotate!(derotate(X, θ), -θ)
    # take some inner indices not wiped out due to cropping
    idxs = Colon(), 250:262, 250:262
    @test_broken X[idxs...] ≈ B[idxs...] rtol=1e-3

    # make sure it mutates
    targ = deepcopy(X)
    derotate!(X, θ)
    @test X !== targ

    # trivial
    X = zeros(1, 3, 3)
    X[1, 1, 2] = 1
    # expect the 1 to rotate 90° ccw
    @test derotate(X, [90])[1, 2, 1] ≈ 1 rtol=1e-3

end

@testset "collapse" begin
    X = rand(10, 512, 512)
    angles = sort!(90 .* rand(10))

    @test collapse(X) == collapse(X; method = median)
    @test collapse(X; method = mean) == mean(X, dims = 1)[1, :, :]

    @test !(collapse(X, angles) ≈ collapse(derotate(X, angles)))
    @test collapse(X, angles, deweight=false) ≈ collapse(derotate(X, angles))
    @test any(collapse(X, angles, fill=NaN) .=== NaN)

    @test collapse(ones(10, 512, 512), angles) == mean(derotate!(ones(10, 512, 512), angles), dims=1)[1, :, :]

end
