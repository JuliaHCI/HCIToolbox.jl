
@testset "scale" begin
    data = ones(2, 2, 3)
    scales = [1, 2, 3]
    sc_cube = scale(data, scales)
    @test size(sc_cube) == (6, 6, 3)
    @test all(sc_cube[3:4, 3:4, 1] .≈ 1)
    @test all(sc_cube[2:5, 2:5, 2] .≈ 1)
    @test all(sc_cube[3, :, :] .≈ 1)
    @test sum(sc_cube, dims=(1, 2)) ≈ [36, 16, 4]

    for i in axes(data, 3)
        @test scale(data[:, :, i], scales[i], (6, 6)) ≈ sc_cube[:, :, i]
    end
end

@testset "invscale" begin
    data = ones(6, 6, 3)
    scales = [1, 2, 3]
    sc_cube = invscale(data, scales)
    @test size(sc_cube) == (2, 2, 3)
    @test all(sc_cube .≈ 1)

    for i in axes(data, 3)
        @test invscale(data[:, :, i], scales[i], (2, 2)) ≈ sc_cube[:, :, i]
    end
end

@testset "stack and collapse" begin
    data = ones(2, 2, 4, 3)
    scales = [1, 2, 3]
    stack = scale_and_stack(data, scales)
    @test size(stack) == (6, 6, 12)
    @test all(stack[3:4, 3:4, 1:3:12] .≈ 1)
    @test all(stack[2:5, 2:5, 2:3:12] .≈ 1)
    @test all(stack[:, :, 3:3:12] .≈ 1)
    @test sum(stack) ≈ 4 * sum((36, 16, 4))

    cube = invscale_and_collapse(stack, scales, (2, 2))
    @test size(cube) == (2, 2, 4)
    @test all(cube .≈ 1)
end

@testset "scale list" begin
    waves = [100, 200, 300]
    @test scale_list(waves) ≈ [3, 1.5, 1]    
end
