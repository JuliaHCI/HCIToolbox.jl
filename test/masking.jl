@testset "Masking" begin
    X = ones(3, 3)
    out = mask_circle(X, 1)
    @test out[2, 2] === NaN
    @test all(i == 5 ? true : out[i] !== NaN for i in eachindex(out))
    @test all(mask_circle(X, 2) .=== NaN)

    # interface
    A = mask_circle(ones(100, 100), 5) 
    B = mask_circle!(ones(100, 100), 5)
    @test filter(!isnan, A) == filter(!isnan, B)
    @test_throws MethodError mask_circle(ones(3, 100, 100), 5)
    @test_throws MethodError mask_circle(ones(Int, 100, 100), 5)

end
