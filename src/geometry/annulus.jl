
struct AnnulusView{T,N,M<:AbstractArray{T,N},IT} <: AbstractArray{T,N}
    parent::M
    rmin::Float64
    rmax::Float64
    indices::IT
    fill::T
end

"""
    AnnulusView(arr::AbstractArray{T,3}; inner=0, outer=last(size(parent)) / 2 + 0.5, fill=0)

Cut out an annulus with inner radius `inner` and outer radius `outer`. Values that fall outside of this region will be replaced with `fill`. This does not copy any data, it is merely a view into the data.
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

Base.parent(view::AnnulusView) = view.parent
Base.size(view::AnnulusView) = size(parent(view))
Base.copy(view::AnnulusView) = AnnulusView(copy(parent(view)), view.rmin, view.rmax, view.indices, view.fill)

inside_annulus(view::AnnulusView, args...) = inside_annulus(view.rmin, view.rmax, center(view)[2:3], args...)

@propagate_inbounds function Base.getindex(view::AnnulusView{T,N}, idx::Vararg{<:Integer,N}) where {T,N}
    @boundscheck checkbounds(parent(view), idx...)
    ifelse(inside_annulus(view, idx...), convert(T, parent(view)[idx...]), view.fill)
end

@propagate_inbounds function Base.setindex!(view::AnnulusView{T,N}, val, idx::Vararg{<:Integer,N}) where {T,N}
    @boundscheck checkbounds(parent(view), idx...)
    if inside_annulus(view, idx...)
        parent(view)[idx...] = val
    else
        view.fill
    end
end

function (view::AnnulusView)(asview=false)
    if asview
        @view parent(view)[view.indices...]
    else
        parent(view)[view.indices...]
    end
end

function Base.copyto!(view::AnnulusView, mat::AbstractMatrix)
    inverse!(view, view.parent, mat)
    return view
end

function inverse!(view::AnnulusView, out, mat)
    out[view.indices...] = mat
    return out
end

function inverse(view::AnnulusView, mat)
    out = fill!(similar(parent(view)), view.fill)
    inverse!(view, out, mat)
end
