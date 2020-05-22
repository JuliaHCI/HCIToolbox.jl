@testset "Kernels - $K" for K in [Kernels.Gaussian, Kernels.Normal, Kernels.Moffat, Kernels.AiryDisk]
    kernel = K(10)

    @test kernel(0) == 1
    @test kernel(Inf) ≈ 0

    @test @inferred(kernel(1.0)) == @inferred(kernel(1))
end

@testset "construct - $K" for K in [Kernels.Gaussian, Kernels.Normal, Kernels.Moffat, Kernels.AiryDisk]
    # using size
    
end
