using Distributions
using ImageTransformations: center
using Optim

abstract type PSFModel{T} end

struct Gaussian{T, D<:MvNormal{T}} <: PSFModel{T}
    A::T
    dist::D
end

Gaussian(x0::T, y0::T, Γ::T, A::T) where T = Gaussian(A, MvNormal([x0, y0], Γ / (2sqrt(2log(2)))))
Gaussian(x0, y0, Γ, A) = Gaussian(promote(x0, y0, Γ, A)...)
evaluate(g::Gaussian, x, y) = g.A * pdf(g.dist, [x, y])

function synth_psf(m::PSFModel{T}, shape) where T
    cy, cx = @. shape ./ 2 + 0.5
    psf = Matrix{T}(undef, shape)
    for idx in CartesianIndices(psf)
        y, x = idx.I
        psf[idx] = evaluate(m, x - cx, y - cy)
    end
    return psf
end

function fit_psf(M::Type{<:PSFModel}, data::AbstractMatrix{T}, shape=size(data)) where T
    loss(X) = mapreduce((y_pred, y) -> (y_pred - y)^2, +, synth_psf(M(X...), shape), data)
    func = TwiceDifferentiable(loss, Float64[0, 0, 1, 1]; autodiff=:forward)
    opt = optimize(func, Float64[0, 0, 1, 1])
    @info opt
    return synth_psf(M(Optim.minimizer(opt)...), shape)
end

