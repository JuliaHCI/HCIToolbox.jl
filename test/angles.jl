@testset "normalize_par_angles" begin
    @test normalize_par_angles([-30, -20, -10]) == [330, 340, 350]
    @test normalize_par_angles([-10, 50, 51]) == [350, 410, 411]
    @test normalize_par_angles([20, 40, 60]) == [20, 40, 60]

    x = [-10, 50, 51]
    normalize_par_angles!(x)
    @test x == [350, 410, 411]
end
