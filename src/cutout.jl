
struct CutoutView{T,N,M<:AbstractArray{T,N},CT,IT} <: AbstractArray{T,N}
    parent::M
    center::CT
    indices::IT
    fill::T
end


function CutoutView(parent::AbstractArray{T}, center, _size; fill=zero(T)) where T
    
    xlims = extrema(axes(parent, 1))
    ylims = extrema(axes(parent, 2))

    half_length = ones(Int, 2) .* _size ./ 2
    minx = max(xlims[begin], round(Int, center[begin] - half_length[begin]))
    miny = max(ylims[begin], round(Int, center[end] - half_length[end]))
    maxx = minx + round(Int, half_length[begin] * 2) - 1
    maxy = miny + round(Int, half_length[end] * 2) - 1
    inds = (minx:maxx, miny:maxy,)

    return CutoutView(parent, center, inds, T(fill))
end

function CutoutView(parent::AbstractArray{T}, _size; fill=zero(T)) where T
    center = HCIToolbox.center(parent)
    xlims = extrema(axes(parent, 1))
    ylims = extrema(axes(parent, 2))

    half_length = ones(Int, 2) .* _size ./ 2
    minx = max(xlims[begin], round(Int, center[begin] - half_length[begin]))
    miny = max(ylims[begin], round(Int, center[end] - half_length[end]))
    maxx = minx + round(Int, half_length[begin] * 2)
    maxy = miny + round(Int, half_length[end] * 2)
    inds = (minx:maxx, miny:maxy,)

    return CutoutView(parent, center, inds, T(fill))
end

Base.parent(view::CutoutView) = view.parent
Base.size(view::CutoutView) = map(length, view.indices)
Base.copy(view::CutoutView) = CutoutView(copy(parent(view)), view.center, view.indices, view.fill)
Base.axes(view::CutoutView) = view.indices


@propagate_inbounds function Base.getindex(view::CutoutView{T,N}, idx::Vararg{<:Integer,N}) where {T,N}
    @boundscheck checkbounds(parent(view), idx...)

    value = convert(T, parent(view)[idx...])
    return value
    # ifelse(inside_annulus(view, idx...), , view.fill)
end

# @propagate_inbounds function Base.setindex!(view::CutoutView{T,N}, val, idx::Vararg{<:Integer,N}) where {T,N}
#     @boundscheck checkbounds(parent(view), idx...)
#     if inside_annulus(view, idx...)
#         parent(view)[idx...] = val
#     else
#         view.fill
#     end
# end
