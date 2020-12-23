using PSFModels: Gaussian, Moffat, AiryDisk

# TODO - figure out why using `ones(1, 1)` fails only in test
@testset "image injection - $kernel" for kernel in [[0 0 0; 0 1 0; 0 0 0],
                                          Gaussian(1e-2),
                                          Moffat(1e-3),
                                          AiryDisk(1e-2)]
    frame = zeros(3, 3)
    @test inject(frame, kernel, (1, 3))[3, 1] ≈ 1
    @test inject(frame, kernel, (2, 2); A=2)[2, 2] ≈ 2
    @test inject(frame, kernel, (10, 10)) ≈ zeros(3, 3) atol = sqrt(eps(eltype(frame)))

    cube = zeros(10, 3, 3)
    @test inject(cube, kernel, (2, 2))[:, 2, 2] ≈ ones(10)
    @test inject(cube, 90ones(10), kernel, (2, 2))[:, 2, 2] ≈ ones(10)
end

@testset "cube generator - $kernel" for kernel in [[0 0 0; 0 1 0; 0 0 0],
                                          Gaussian(1e-2),
                                          Moffat(1e-3),
                                          AiryDisk(1e-2)]
    cube = zeros(10, 3, 3)
    angles = 90ones(10)
    gen = CubeGenerator(cube, angles, kernel)

    c = gen((2, 2))
    c2 = gen(Float32, (2, 2))
    c3 = gen(Polar(0, 0))
    @test c ≈ c2 ≈ c3
    @test eltype(c2) === Float32

    q = gen(ones(size(cube)), (2, 2))
    @test all(q .≈ c .+ 1)

    flat = gen(zero(flatten(cube)), (2, 2))
    @test flat ≈ flatten(c)
end
