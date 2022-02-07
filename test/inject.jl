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

@testset "inject - AnnulusView" begin
    cube = AnnulusView(zeros(21, 21, 10))
    angles = 90 .* ones(10)

    kernel = [0 0 0; 0 1 0; 0 0 0]
    c = inject(cube, kernel, angles; x=11, y=11)
    @test c isa AnnulusView
    @test all(c[11, 11, :] .≈ 1)

    @testset "synthetic psf - $mod" for mod in [gaussian, moffat, airydisk]
        c = inject(cube, mod, angles; x=11, y=11, fwhm=3)
        @test c isa AnnulusView
        @test all(c[11, 11, :] .≈ 1)
    end
end

@testset "inject - MultiAnnulusView" begin
    cube = MultiAnnulusView(zeros(21, 21, 10), 5)
    angles = 90 .* ones(10)

    kernel = [0 0 0; 0 1 0; 0 0 0]
    c = inject(cube, kernel, angles; x=11, y=11)
    @test c isa MultiAnnulusView
    @test all(c[11, 11, :] .≈ 1)

    @testset "synthetic psf - $mod" for mod in [gaussian, moffat, airydisk]
        c = inject(cube, mod, angles; x=11, y=11, fwhm=3)
        @test c isa MultiAnnulusView
        @test all(c[11, 11, :] .≈ 1)
    end
end
