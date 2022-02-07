@testset "matrix/cube" begin

    @testset "flatten" begin
        @test flatten(ones(4, 4, 3)) == ones(16, 3)
        @test flatten(ones(4, 3)) == ones(4, 3)
    end

    @testset "expand" begin
        @test expand(ones(16, 3)) == ones(4, 4, 3)
        @test expand(ones(4, 4, 3)) == ones(4, 4, 3)

        @test_throws ErrorException expand(ones(15, 3))
    end

    # Simple regression
    X = rand(rng, 512, 512, 10)
    @test expand(flatten(X)) == X
end

@testset "derotate" begin
    X = rand(rng, 512, 512, 10)
    
    # test no-op
    @test derotate(X, zeros(10)) === X

    # Simple regression (but be careful to only test some inner part that won't be clipped)
    X = rand(rng, 512, 512, 10)
    θ = 90 .* rand(rng, 10)
    B = derotate!(derotate(X, θ), -θ)
    # take some inner indices not wiped out due to cropping
    idxs = 250:262, 250:262, Colon()
    @test_broken X[idxs...] ≈ B[idxs...] rtol=1e-3

    # make sure it mutates
    targ = deepcopy(X)
    derotate!(X, θ)
    @test X !== targ

    # trivial
    X = zeros(3, 3, 1)
    X[2, 1, 1] = 1
    # expect the 1 to rotate 90° ccw physical, cw image
    @test derotate(X, [-90])[1, 2, 1] ≈ 1 rtol=1e-3

end

@testset "collapse" begin
    X = rand(rng, 512, 512, 10)
    angles = sort!(90 .* rand(rng, 10))

    @test collapse(X) == collapse(X; method = median)
    @test collapse(X; method = mean) == mean(X, dims = 3)[:, :, 1]

    @test !(collapse(X, angles) ≈ collapse(derotate(X, angles)))
    @test collapse(X, angles, method=median) ≈ collapse(derotate(X, angles))
    @test any(collapse(X, angles, fill=NaN) .=== NaN)

    @test collapse(ones(512, 512, 10), angles) == mean(derotate!(ones(512, 512, 10), angles), dims=3)[:, :, 1]

    X_copy = deepcopy(X)
    @test collapse(X, angles) == collapse!(X_copy, angles)
    @test X_copy != X

end

@testset "shift frame" begin
    X = zeros(3, 3)
    X[2, 2] = 1
    for dx in -1:1, dy in -1:1
        row = 2 + dx
        col = 2 + dy
        @test shift_frame(X, dx, dy)[row, col] == 1
        @test shift_frame(X, (dx, dy)) == shift_frame(X, dx, dy)
    end

    @test shift_frame(X, 1, 1, fill=NaN)[3, 1] === NaN

    # cube
    cube = zeros(5, 5, 10)
    cube[3, 3, :] .= 1
    for dx in -1:1, dy in -1:1
        row = 3 + dx
        col = 3 + dy
        @test shift_frame(cube, dx, dy)[row, col, :] == ones(10)
        shift_frame(cube, (dx, dy)) == shift_frame(cube, dx, dy)
        shift_frame(cube, fill(dx, 10), fill(dy, 10)) == shift_frame(cube, dx, dy)
        shift_frame(cube, fill((dx, dy), 10)) == shift_frame(cube, dx, dy)
    end
    @test all(shift_frame(cube, 1, 1, fill=NaN)[1, 3, :] .=== NaN)
end

@testset "crop & cropview" begin
    cube = zeros(101, 101, 10)

    @test crop(cube, 51) == crop(cube, (51, 51)) == cube[26:76, 26:76, :]
    @test crop(cube, 51, center=(52, 50)) == cube[27:77, 25:75, :]

    @test size(crop(cube, 51)) == (51, 51, 10)
    @test_logs (:info, "adjusted size to (51, 51) to avoid uneven (odd) cropping") begin
        @test size(crop(cube, 50)) == (51, 51, 10)
    end
    @test size(crop(cube, 50; force=true)) == (50, 50, 10)

    cube_crop_view = cropview(cube, 51)
    @test parent(cube_crop_view) === cube
    @test collect(cube_crop_view) == Array(cube_crop_view) == crop(cube, 51)

    ### frames

    frame = zeros(101, 101)

    @test crop(frame, 51) == crop(frame, (51, 51)) == frame[26:76, 26:76]
    @test crop(frame, 51, center=(52, 50)) == frame[27:77, 25:75]

    @test size(crop(frame, 51)) == (51, 51)
    @test_logs (:info, "adjusted size to (51, 51) to avoid uneven (odd) cropping") begin
        @test size(crop(frame, 50)) == (51, 51)
    end
    @test size(crop(frame, 50; force=true)) == (50, 50)

    frame_crop_view = cropview(frame, 51)
    @test parent(frame_crop_view) === frame
    @test collect(frame_crop_view) == Array(frame_crop_view) == crop(frame, 51)
end
