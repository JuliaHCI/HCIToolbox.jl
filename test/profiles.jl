
@testset "radial_profile" begin
    
    X = [0 1 1 1 0
         1 2 2 2 1
         1 2 3 2 1
         1 2 2 2 1
         0 1 1 1 0];

    r, prof = @inferred radial_profile(X);
    @test r == [0, 1, 2, 3]
    @test prof ≈ [3, 2, 1, 0]


    X = [1 2
         2 2]
    r, prof = @inferred radial_profile(X, center=(1, 1))
    @test r == [0, 1]
    @test prof ≈ [1, 2]

    img = randn(rng, 1001, 1001)
    r, prof = @inferred radial_profile(img)
    @test mean(prof) ≈ 0 atol=1e-2
end