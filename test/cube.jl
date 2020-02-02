using Statistics

fake_cube_rand = DataCube(rand(100, 100, 3), [0, 10, 20])

fake_cube_ones = DataCube(ones(100, 100, 3), [0, 10, 20])

@testset "DataCube" begin
    no_angles = DataCube(ones(100, 100, 3))
    @test angles(no_angles) == zeros(3)

end

@testset "Cube Shaping" begin
    # flatten and unflatten
    @assert DataCube(Matrix(fake_cube_rand), angles(fake_cube_rand)) == fake_cube_rand
end

@testset "Merging" begin
    @test mean(fake_cube_ones) == ones(100, 100)
    @test median(fake_cube_ones) == ones(100, 100)
end