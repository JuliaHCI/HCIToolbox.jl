@testset "Masking" begin
    @testset "circle" begin
        X = ones(3, 3)
        out = mask_circle(X, 1, fill=NaN)
        @test out[2, 2] === NaN
        @test all(i == 5 ? true : out[i] !== NaN for i in eachindex(out))
        @test all(mask_circle(X, 2, fill=NaN) .=== NaN)

        # interface
        A = @inferred mask_circle(ones(100, 100), 5, fill=NaN)
        B = @inferred mask_circle!(ones(100, 100), 5, fill=NaN)
        @test filter(!isnan, A) == filter(!isnan, B)

        # cubes
        A = @inferred mask_circle(ones(3, 100, 100), 5, fill=NaN)
        B = @inferred mask_circle!(ones(3, 100, 100), 5, fill=NaN)
        @test filter(!isnan, A) == filter(!isnan, B)


    end

    @testset "annulus" begin
        X = ones(3, 3)
        out = mask_annulus(X, 0, 1, fill=NaN)
        @test out[2, 2] === NaN
        @test all(i == 5 ? true : out[i] !== NaN for i in eachindex(out))
        @test all(mask_annulus(X, 0, 2, fill=NaN) .=== NaN)

        # interface
        A = @inferred mask_annulus(ones(100, 100), 3, 6, fill=NaN)
        B = @inferred mask_annulus!(ones(100, 100), 3, 6, fill=NaN)
        @test filter(!isnan, A) == filter(!isnan, B)
        @test_throws MethodError mask_annulus!(ones(3, 100, 100), 5)
    end

    # regression check: annulus with inner 0 should be same as circle
    data = ones(100, 100)
    @test filter(!isnan, mask_circle(data, 10, fill=NaN)) == filter(!isnan, mask_annulus(data, 0, 10, fill=NaN))
end
