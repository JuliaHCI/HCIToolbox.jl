using StaticArrays
using CoordinateTransformations

struct AnnulusView{T,N,M<:AbstractArray{T,N},IT} <: AbstractArray{T,N}
    parent::M
    rmin::Float64
    rmax::Float64
    indices::IT
    fill::T
end

"""
    AnnulusView(arr::AbstractArray{T,3}; inner=0, outer=last(size(parent)) / 2 + 0.5, fill=0)
"""
function AnnulusView(parent::AbstractArray{T,3};
                     inner=0,
                     outer=(size(parent, 3) + 1) / 2,
                     fill=zero(T)) where T
    # check inputs
    0 ≤ inner < outer || error("Invalid annulus region [$inner, $outer]")
    time_axis = axes(parent, 1)
    space_indices = CartesianIndices((axes(parent, 2), axes(parent, 3)))
    space_axis = filter(idx -> inside_annulus(inner, outer, center(parent)[2:3], idx), space_indices)
    return AnnulusView(parent, Float64(inner), Float64(outer), (time_axis, space_axis), T(fill))
end

"""
    AnnulusView(arr::AbstractArray{T,3}, angles; inner=0, outer=last(size(parent))/2+0.5, fill=0, threshold=0, fwhm=0, radius=inner+fwhm/2)
"""
function AnnulusView(parent::AbstractArray{T,3},
                     angles;
                     inner=0,
                     outer=(size(parent, 3) + 1) / 2,
                     fill=zero(T),
                     threshold=0,
                     fwhm=0,
                     radius=(outer - inner)/2 + inner) where T
    # check inputs
    0 ≤ inner < outer || error("Invalid annulus region [$inner, $outer]")
    size(parent, 1) == size(angles, 1) || error("cube and angles do not have same number of frames")

    # minimin rotation between frames
    pa_thresh = rad2deg(2 * atan(threshold * fwhm / (2 * radius)))
    mid_range = abs(maximum(angles) - minimum(angles))
    if pa_thresh ≥ mid_range * 0.9
        old_pa_thresh = pa_thresh
        pa_thresh = mid_range * 0.9
        @info "PA threshold was too large, setting to $pa_thresh"
    end

    time_axis = [firstindex(angles)]
    for idx in Iterators.rest(axes(angles, 1), 2)
        prev = angles[time_axis[end]]
        if abs(angles[idx] - prev) ≥ pa_thresh
            push!(time_axis, idx)
        end
    end

    space_indices = CartesianIndices((axes(parent, 2), axes(parent, 3)))
    space_axis = filter(idx -> inside_annulus(inner, outer, center(parent)[2:3], idx), space_indices)
    
    return AnnulusView(parent, Float64(inner), Float64(outer), (time_axis, space_axis), T(fill))
end

Base.parent(view::AnnulusView) = view.parent
Base.size(view::AnnulusView) = size(parent(view))
Base.copy(view::AnnulusView) = AnnulusView(copy(parent(view)), view.rmin, view.rmax, view.indices, view.fill)

inside_annulus(view::AnnulusView, args...) = inside_annulus(view.rmin, view.rmax, center(view)[2:3], args...)
inside_annulus(rmin, rmax, center, idx...) = inside_annulus(rmin, rmax, center, SVector(idx[end-1:end]))
inside_annulus(rmin, rmax, center, idx::CartesianIndex{2}) = inside_annulus(rmin, rmax, center, SVector(idx.I))
function inside_annulus(rmin, rmax, center::AbstractVector, position::AbstractVector)
    Δ = center - position
    r = sqrt(sum(abs2, Δ))
    return rmin ≤ r ≤ rmax
end

Base.@propagate_inbounds function Base.getindex(view::AnnulusView{T,N}, idx::Vararg{<:Integer,N}) where {T,N}
    @boundscheck checkbounds(parent(view), idx...)
    ifelse(inside_annulus(view, idx...), convert(T, parent(view)[idx...]), view.fill)
end

Base.@propagate_inbounds function Base.setindex!(view::AnnulusView{T,N}, val, idx::Vararg{<:Integer,N}) where {T,N}
    @boundscheck checkbounds(parent(view), idx...)
    if inside_annulus(view, idx...)
        parent(view)[idx...] = val
    else
        view.fill
    end
end

function flatten(view::AnnulusView)
    dims = map(length, view.indices)
    output = similar(parent(view), dims...)
    @inbounds for idx in CartesianIndices(output)
        tidx, pidx = Tuple(idx)
        i = view.indices[1][tidx]
        j = view.indices[2][pidx]
        output[idx] = view.parent[i, j]
    end
    return output
end

function Base.copyto!(view::AnnulusView, mat::AbstractMatrix)
    inverse!(view, view.parent, mat)
    return view
end


function inverse!(view::AnnulusView, out, mat)
    @inbounds for idx in CartesianIndices(mat)
        tidx, pidx = Tuple(idx)
        i = view.indices[1][tidx]
        j = view.indices[2][pidx]
        out[i, j] = mat[idx]
    end
    return out
end

inverse(view::AnnulusView, mat) = inverse!(view, fill!(similar(parent(view)), view.fill), mat)
