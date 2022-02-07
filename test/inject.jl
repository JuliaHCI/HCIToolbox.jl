using PSFModels

# TODO - figure out why using `ones(1, 1)` fails only in test
@testset "image injection" begin
    kernel = [0 0 0; 0 1 0; 0 0 0]
    frame = zeros(3, 3)
    @test inject(frame, kernel; x=1, y=3)[1, 3] ≈ 1
    @test inject(frame, kernel; x=2, y=2, amp=2)[2, 2] ≈ 2
    @test inject(frame, kernel; x=10, y=10) ≈ zeros(3, 3) atol = sqrt(eps(eltype(frame)))

    cube = zeros(3, 3, 10)
    angles = 90 .* ones(10)
    @test inject(cube, kernel; x=2, y=2)[2, 2, :] ≈ ones(10)
    @test inject(cube, kernel, angles; x=2, y=2)[2, 2, :] ≈ ones(10)

    @testset "synthetic psf - $mod" for mod in [gaussian, moffat, airydisk]
        frame = zeros(3, 3)
        @test inject(frame, mod; x=1, y=3, fwhm=3)[1, 3] ≈ 1
        @test inject(frame, mod; x=2, y=2, fwhm=3, amp=2)[2, 2] ≈ 2

        cube = zeros(3, 3, 10)
        @test inject(cube, mod; x=2, y=2, fwhm=3)[2, 2, :] ≈ ones(10)
        @test inject(cube, mod, angles; x=2, y=2, fwhm=3)[2, 2, :] ≈ ones(10)
    end
end

# @testset "inject av - $(typeof(kernel))" for kernel in [[0 0 0; 0 1 0; 0 0 0], Gaussian(1e-2)]
#     cube = AnnulusView(zeros(10, 21, 21))
#     angles = 90ones(10)

#     c = inject(cube, kernel, angles; x=11, y=11)
#     c2 = inject(cube, kernel, angles; x=11, y=11)
#     @test c ≈ c2
#     @test c isa AnnulusView && c2 isa AnnulusView

#     gen = CubeGenerator(cube, angles, kernel)
#     flat = gen(cube(), (11, 11))
#     @test flat ≈ c()
# end

# @testset "inject mav - $(typeof(kernel))" for kernel in [[0 0 0; 0 1 0; 0 0 0], Gaussian(1e-2)]
#     cube = MultiAnnulusView(zeros(10, 21, 21), 2)
#     angles = 90ones(10)

#     c = inject(cube, angles, kernel, (11, 11))
#     c2 = inject(cube, angles, kernel, Polar(0, 0))
#     @test c ≈ c2
#     @test c isa MultiAnnulusView && c2 isa MultiAnnulusView
# end
