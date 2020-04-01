@testset "Masking" begin
    X = ones(3, 3)
    out = mask_circle(X, 1)
    @test out[2, 2] === NaN
    @test all(i == 5 ? true : out[i] !== NaN for i in eachindex(out))
    @test all(mask_circle(X, 2) .=== NaN)

    X = ones(3, 3)
    out = mask_annulus(X, 0, 1)
    @test out[2, 2] === NaN
    @test all(i == 5 ? true : out[i] !== NaN for i in eachindex(out))
    @test all(mask_annulus(X, 0, 2) .=== NaN)

    # interface
    A = mask_circle(ones(100, 100), 5)
    B = mask_circle!(ones(100, 100), 5)
    @test filter(!isnan, A) == filter(!isnan, B)
    @test_throws MethodError mask_circle(ones(3, 100, 100), 5)
    @test_throws MethodError mask_circle(ones(Int, 100, 100), 5)

    A = mask_annulus(ones(100, 100), 3, 6)
    B = mask_annulus!(ones(100, 100), 3, 6)
    @test filter(!isnan, A) == filter(!isnan, B)
    @test_throws MethodError mask_annulus!(ones(3, 100, 100), 5)
    @test_throws MethodError mask_annulus(ones(Int, 100, 100), 5)
end
