
struct MultiAnnulusView{T,N,M<:AbstractArray{T,N},RT,IT} <: AbstractArray{T,N}
    parent::M
    radii::RT
    width::Float64
    indices::IT
    fill::T
end


"""
    MultiAnnulusView(cube::AbstractArray{T,3} width, radii; fill=0)

Create multiple annuli at each radius in `radii` with width `width`. Values that fall outside of these regions will be replaced with `fill`. This does not copy any data, it is merely a view into the data.
"""
function MultiAnnulusView(parent::AbstractArray{T,3}, width, radii; fill=zero(T)) where T
    time_axis = axes(parent, 1)
    space_indices = CartesianIndices((axes(parent, 2), axes(parent, 3)))
    space_axes = map(radii) do r
        inner = r - width / 2
        outer = r + width / 2
        filter(idx -> inside_annulus(inner, outer, center(parent)[2:3], idx), space_indices)
    end
    indices = [(time_axis, space_axis) for space_axis in space_axes]
    return MultiAnnulusView(parent, radii, Float64(width), indices, convert(T, fill))
end

"""
    MultiAnnulusView(cube::AbstractArray{T,3}, width;
                     inner=0, outer=last(size(parent))/2 + 0.5,
                     fill=0)

Create multiple annuli between `inner` and `outer` with `width` spacing. Values that fall outside of these regions will be replaced with `fill`. This does not copy any data, it is merely a view into the data.
"""
function MultiAnnulusView(parent::AbstractArray{T,3}, width; inner=0, outer=(last(size(parent)) + 1) / 2, fill=zero(T)) where T
    first_r = inner + width/2
    if isfinite(outer)
        final_r = outer - width/2
    else
        max_length = (last(size(parent)) + 1) / 2
        max_r = sqrt(2 * max_length^2)
        final_r = max_r + 1 / sqrt(2)
    end
    radii = range(first_r, final_r, step=width)
    return MultiAnnulusView(parent, width, radii; fill=fill)
end

Base.parent(view::MultiAnnulusView) = view.parent
Base.size(view::MultiAnnulusView) = size(parent(view))
Base.copy(view::MultiAnnulusView) = MultiAnnulusView(copy(parent(view)), view.radii, view.width, view.indices, view.fill)

@propagate_inbounds function Base.getindex(view::MultiAnnulusView{T,N}, idx::Vararg{<:Integer,N}) where {T,N}
    @boundscheck checkbounds(parent(view), idx...)
    inside = any(r -> inside_annulus(r-view.width/2, r+view.width/2, center(parent(view))[2:3], idx...), view.radii)
    ifelse(inside, convert(T, parent(view)[idx...]), view.fill)
end

@propagate_inbounds function Base.setindex!(view::MultiAnnulusView{T,N}, val, idx::Vararg{<:Integer,N}) where {T,N}
    @boundscheck checkbounds(parent(view), idx...)
    inside = any(r -> inside_annulus(r - view.width/2, r + view.width/2, center(parent(view))[2:3], idx...), view.radii)
    if inside
        parent(view)[idx...] = val
    else
        view.fill
    end
end

"""
    (::MultiAnnulusView)(idx, asview=false)

Return the `idx`th annulus as a matrix. This is equivalent to unrolling the frame and filtering out pixels outside of the `idx`th annulus. If `asview` is true, the returned values will be a view of the parent array instead of a copy.

# See also
[`eachannulus`](@ref)
"""
@propagate_inbounds function (view::MultiAnnulusView)(idx::Int, asview=false)
    @boundscheck checkbounds(view.indices, idx)
    idxs = view.indices[idx]
    if asview
        @view parent(view)[idxs...]
    else
        parent(view)[idxs...]
    end
end

"""
    eachannulus(::MultiAnnulusView, asview=false)
"""
eachannulus(view::MultiAnnulusView, asview=false) = (view(i, asview) for i in 1:length(view.indices))

function Base.copyto!(view::MultiAnnulusView, mats::Vararg{<:AbstractMatrix})
    if length(mats) != length(view.indices)
        error("Number of matrices does not match number of annuli")
    end
    inverse!(view, view.parent, mats...)
    return view
end

Base.copyto!(view::MultiAnnulusView, mats::AbstractVector{<:AbstractMatrix}) = copyto!(view, mats...)

function Base.copyto!(view::MultiAnnulusView, idx::Int, mat::AbstractMatrix)
    inverse!(view, view.parent, idx, mat)
    return view
end

function inverse!(view::MultiAnnulusView, out, mats::Vararg{<:AbstractMatrix})
    if length(mats) != length(view.indices)
        error("Number of matrices does not match number of annuli")
    end
    for (idxs, mat) in zip(view.indices, mats)
        out[idxs...] = mat
    end
    return out
end

function inverse!(view::MultiAnnulusView, out, idx::Int, mat::AbstractMatrix)
    idxs = view.indices[idx]
    out[idxs...] = mat
    return out
end

inverse!(view::MultiAnnulusView, out, mats) = inverse!(view, out, mats...)

function inverse(view::MultiAnnulusView, args...)
    out = fill!(similar(parent(view)), view.fill)
    inverse!(view, out, args...)
end
inverse(view::MultiAnnulusView, mats) = inverse(view, mats...)
