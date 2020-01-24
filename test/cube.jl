using Statistics

fake_cube_rand = DataCube(rand(100, 100, 3), [0, 10, 20])

fake_cube_ones = DataCube(ones(100, 100, 3), [0, 10, 20])

@testset "DataCube" begin

end

@testset "Merging" begin
    @test mean(fake_cube_ones) == ones(100, 100)
    @test median(fake_cube_ones) == ones(100, 100)
end