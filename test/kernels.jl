@test Kernels.Normal === Kernels.Gaussian
@test string(@doc(Kernels.Normal)) == string(@doc(Kernels.Gaussian))

@testset "Kernels - $K" for K in [Kernels.Gaussian, Kernels.Moffat, Kernels.AiryDisk]
    kernel = K(10)

    @test kernel(0) ≈ 1
    @test kernel(Inf) ≈ 0

    @test @inferred(kernel(1.0)) == @inferred(kernel(1))
end

@testset "construct - $kernel" for kernel in [Kernels.Gaussian(10), Kernels.Moffat(3), Kernels.AiryDisk(10)]
    # using size
    arr = @inferred construct(kernel, (101, 101); r=0, θ=0)
    @test size(arr) == (101, 101)
    @test maximum(arr) ≈ 1
    @test minimum(arr) ≈ 0 atol=1e-3 # atol because Moffat does NOT reduce quickly

    # using indices
    @test arr == @inferred construct(kernel, (1:101, 1:101); r=0, θ=0)

    # location parsing
    @test construct(kernel, (101, 101); A=2, r=3, θ=0) ≈ construct(kernel, (101, 101); A=2, x=54, y=51)
end
