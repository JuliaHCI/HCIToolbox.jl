using StaticArrays
using CoordinateTransformations

struct AnnulusView{T,N,M<:AbstractArray{T,N},IT} <: AbstractArray{T,N}
    parent::M
    rmin::Float64
    rmax::Float64
    indices::IT
    fill::T
end

function AnnulusView(parent::AbstractArray{T,3};
                     inner=0,
                     outer=(size(parent, 3) + 1) / 2,
                     fill=zero(T)) where T
    # check inputs
    0 ≤ inner < outer || error("Invalid annulus region [$inner, $outer]")
    time_axis = axes(parent, 1)
    space_axis = Iterators.filter(idx -> inside_annulus(inner, outer, center(parent), idx), eachindex(parent))
    return AnnulusView(parent, Float64(inner), Float64(outer), (time_axis, space_axis), T(fill))
end

Base.parent(view::AnnulusView) = view.parent
Base.size(view::AnnulusView) = size(parent(view))


inside_annulus(rmin, rmax, center, idx...) = inside_annulus(rmin, rmax, center, SVector(idx...))
inside_annulus(rmin, rmax, center, idx::CartesianIndex) = inside_annulus(rmin, rmax, center, SVector(Tuple(idx)))
inside_annulus(view::AnnulusView, args...) = inside_annulus(view.rmin, view.rmax, center(view), args...)
function inside_annulus(rmin, rmax, center, position::AbstractVector)
    Δ = center[end-1:end] - position[end-1:end]
    r = sqrt(sum(abs2, Δ))
    return rmin ≤ r ≤ rmax
end

function Base.getindex(view::AnnulusView, idx...)
    return inside_annulus(view, idx...) ? parent(view)[idx...] : view.fill
end

function Base.getindex(view::AnnulusView, tidx, pos...)
    return inside_annulus(view, pos...) ? parent(view)[tidx, pos...] : view.fill
end

function Base.setindex!(view::AnnulusView, val, idx...)
    if inside_annulus(view, idx...)
        parent(view)[idx...] = val
    else
        view.fill
    end
end

function Base.setindex!(view::AnnulusView, val, tidx, pos...)
    if inside_annulus(view, pos...)
        parent(view)[tidx, pos...] = val
    else
        view.fill
    end
end

function flatten(view::AnnulusView, ::Val{true})
    map(Iterators.product(view.indices...)) do (tidx, pidx)
        pidx′ = CartesianIndex(pidx.I[2:3])
       parent(view)[tidx, pidx′]
    end
end

Base.copy(view::AnnulusView) = AnnulusView(copy(view.parent), view.rmin, view.rmax, view.indices, view.fill)

function Base.copyto!(view::AnnulusView, mat::AbstractMatrix)
    for (tidx, tidx′) in zip(axes(mat, 1), view.indices[1])
        for (pidx, pidx′) in zip(axes(mat, 2), view.indices[2])
            @inbounds view[tidx′, pidx′] = mat[tidx, pidx]
        end
    end
    return view
end

function med(view::AnnulusView, angles)
    X = flatten(view, Val(true))
    R = X .- median(X, dims=1)
    out = copyto!(copy(view), R)
    return collapse(out, angles)
end
