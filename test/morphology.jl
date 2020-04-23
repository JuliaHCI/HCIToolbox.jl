@testset "matrix/cube" begin

    @testset "tomatrix" begin
        @test tomatrix(ones(3, 4, 4)) == ones(3, 16)

        @test_throws MethodError tomatrix(ones(3, 4))
    end

    @testset "tocube" begin
        @test tocube(ones(3, 16)) == ones(3, 4, 4)

        @test_throws ErrorException tocube(ones(3, 15))
    end

    # Simple regression
    X = rand(10, 512, 512)
    @test tocube(tomatrix(X)) == X
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

@testset "shift frame" begin
    X = zeros(3, 3)
    X[2, 2] = 1
    for dx in -1:1, dy in -1:1
        row = 2 + dy
        col = 2 + dx
        @test shift_frame(X, dx, dy)[row, col] == 1
    end

    @test shift_frame(X, 1, 1, fill=NaN)[3, 1] === NaN
    
end

# TODO WHY IS THIS ALL BROKEN???????
@testset "image injection" begin
    frame = zeros(3, 3)
    img = ones(1, 1)
    for x in 1:3, y in 1:3
        @test inject_image(frame, img, x = x, y = y)[y, x] == 1
        @test inject_image(frame, img, A=2, x = x, y = y)[y, x] == 2
    end
    @test inject_image(frame, img, x = 0, y = 0) == zeros(3, 3)

    ## the following is not G2G
    # for (x, y) in zip([-1, 0, 0, 1], [0, 1, -1, 0]), frame in [zeros(3, 3), zeros(4, 4)], img in [ones(1, 1), ones(2, 2)]
    #     # works in both coordinate systems
    #     r = hypot(x, y)
    #     theta = atan(y, x)
    #     @test inject_image(frame, img; x=x, y=y) ≈ inject_image(frame, img; r=r, theta=theta)
    #     # kwarg position invariance
    #     @test inject_image(frame, img; x=x, y=y) == inject_image(frame, img; y=y, x=x)
    #     @test inject_image(frame, img; r=r, theta=theta) == inject_image(frame, img; theta=theta, r=r)
    # end
end
