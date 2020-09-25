
@testset "scale" begin
    data = ones(3, 2, 2)
    scales = [1, 2, 3]
    sc_cube = scale(data, scales)
    @test size(sc_cube) == (3, 6, 6)
    @test all(sc_cube[1, 3:4, 3:4] .≈ 1)
    @test all(sc_cube[2, 2:5, 2:5] .≈ 1)
    @test all(sc_cube[3, :, :] .≈ 1)
    @test sum(sc_cube, dims=(2, 3)) ≈ [4, 16, 36]

    for i in axes(data, 1)
        @test scale(data[i, :, :], scales[i], (6, 6)) ≈ sc_cube[i, :, :]
    end
end

@testset "invscale" begin
    data = ones(3, 6, 6)
    scales = [1, 2, 3]
    sc_cube = invscale(data, scales)
    @test size(sc_cube) == (3, 2, 2)
    @test all(sc_cube .≈ 1)

    for i in axes(data, 1)
        @test invscale(data[i, :, :], scales[i], (2, 2)) ≈ sc_cube[i, :, :]
    end
end

@testset "stack and collapse" begin
    data = ones(3, 4, 2, 2)
    scales = [1, 2, 3]
    stack = scale_and_stack(data, scales)
    @test size(stack) == (12, 6, 6)
    @test all(stack[1:3:12, 3:4, 3:4] .≈ 1)
    @test all(stack[2:3:12, 2:5, 2:5] .≈ 1)
    @test all(stack[3:3:12, :, :] .≈ 1)
    @test sum(stack) ≈ 4 * sum((4, 16, 36))

    cube = invscale_and_collapse(stack, scales, (2, 2))
    @test size(cube) == (4, 2, 2)
    @test all(cube .≈ 1)
end

@testset "scale list" begin
    waves = [100, 200, 300]
    @test scale_list(waves) ≈ [3, 1.5, 1]    
end
