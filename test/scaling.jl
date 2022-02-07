
@testset "scale" begin
    data = ones(2, 2, 3)
    scales = [3, 2, 1]
    sc_cube = scale(data, scales)
    @test size(sc_cube) == size(data)
    @test all(sc_cube .≈ 1)

    for i in axes(data, 3)
        @test scale(data[:, :, i], scales[i]) ≈ sc_cube[:, :, i]
    end
end

@testset "invscale" begin
    data = ones(8, 8, 3)
    scales = [4, 2, 1]
    sc_cube = invscale(data, scales)
    @test size(sc_cube) == size(data)
    @test all(sc_cube[4:5, 4:5, 1] .≈ 1)
    @test all(sc_cube[3:6, 3:6, 2] .≈ 1)
    @test all(sc_cube[:, :, 3] .≈ 1)
    @test sum(sc_cube, dims=(1, 2)) |> vec ≈ [4, 16, 64]

    for i in axes(data, 3)
        @test invscale(data[:, :, i], scales[i]) ≈ sc_cube[:, :, i]
    end
end

@testset "stack and collapse" begin
    data = ones(8, 8, 3, 4)
    scales = [4, 2, 1]
    stack = scale_and_stack(data, scales)
    @test size(stack) == (8, 8, 12)
    @test all(stack .≈ 1)
    @test sum(stack) ≈ length(data)

    cube = invscale_and_collapse(stack, scales; method=sum)
    @test size(cube) == (8, 8, 4)
    @test sum(cube) ≈ 336
end

@testset "scale list" begin
    waves = [100, 200, 300]
    @test scale_list(waves) ≈ [3, 1.5, 1]    
end
