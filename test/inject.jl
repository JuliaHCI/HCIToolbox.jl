@test Kernels.Normal === Kernels.Gaussian
@test string(@doc(Kernels.Normal)) == string(@doc(Kernels.Gaussian))

@testset "Kernels - $K" for K in [Kernels.Gaussian, Kernels.Moffat, Kernels.AiryDisk]
    kernel = K(10)

    @test kernel(0) ≈ 1
    @test kernel(Inf) ≈ 0

    @test @inferred(kernel(1.0)) == @inferred(kernel(1))
end

@testset "construct - $kernel" for kernel in [Kernels.Gaussian(10), Kernels.Moffat(3), Kernels.AiryDisk(10)]
    # TODO - inferred tests are failing here on 1.3
    # using size
    arr = construct(kernel, (101, 101); r=0, θ=0)
    @test size(arr) == (101, 101)
    @test maximum(arr) ≈ 1
    @test minimum(arr) ≈ 0 atol=1e-3 # atol because Moffat does NOT reduce quickly

    # using indices
    @test arr == construct(kernel, (1:101, 1:101); r=0, θ=0)

    # location parsing
    A = construct(kernel, (101, 101); A=2, r=3, θ=0)
    B = construct(kernel, (101, 101); A=2, x=54, y=51)
    @test A ≈ B 
    @test A[51, 54] ≈ B[51, 54] ≈ 2

    A = construct(kernel, (101, 101); A=2, r=3, θ=0, pa=90)
    B = construct(kernel, (101, 101); A=2, x=54, y=51, pa=90)
    @test A ≈ B 
    @test A[48, 51] ≈ B[48, 51] ≈ 2
end

# TODO - figure out why using `ones(1, 1)` fails only in test
@testset "image injection - $kernel" for kernel in [[0 0 0; 0 1 0; 0 0 0],
                                          Kernels.Gaussian(1),
                                          Kernels.Moffat(1e-3),
                                          Kernels.AiryDisk(1e-2)]
    frame = zeros(3, 3)
    @test inject(frame, kernel; x = 1, y = 3)[3, 1] ≈ 1
    @test inject(frame, kernel; A=2, x = 2, y = 2)[2, 2] ≈ 2
    @test inject(frame, kernel; x = 10, y = 10) ≈ zeros(3, 3) atol = sqrt(eps(eltype(frame)))

    cube = zeros(10, 3, 3)
    @test inject(cube, kernel; x=2, y=2)[:, 2, 2] ≈ ones(10)
    @test inject(cube, kernel, zeros(10); x=2, y=2)[:, 2, 2] ≈ ones(10)

    @test_throws ErrorException inject(cube, kernel, zeros(3); x=2, y=2)
end
