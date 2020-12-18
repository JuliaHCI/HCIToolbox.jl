
struct MultiAnnulusView{T,N,M<:AbstractArray{T,N},RT,IT} <: AbstractArray{T,N}
    parent::M
    radii::RT
    width::Float64
    indices::IT
    fill::T
end

function MultiAnnulusView(parent::AbstractArray{T,3}, radii, width; fill=zero(T)) where T
    time_axis = axes(parent, 1)
    space_indices = CartesianIndices((axes(parent, 2), axes(parent, 3)))
    space_axes = map(radii) do r
        inner = r - width / 2
        outer = r + width / 2
        filter(idx -> inside_annulus(inner, outer, center(parent)[2:3], idx), space_indices)
    end
    indices = [(time_axis, space_axis) for space_axis in space_axes]
    return MultiAnnulusView(parent, radii, width, indices, convert(T, fill))
end

function MultiAnnulusView(parent::AbstractArray{T,3}, width; inner=0, outer=(last(size(parent)) + 1) / 2, fill=zero(T)) where T
    first_r = inner + width/2
    final_r = outer - width/2
    radii = first_r:width:final_r
    return MultiAnnulusView(parent, radii, width; fill=fill)
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

@propagate_inbounds function (view::MultiAnnulusView)(idx::Int)
    @boundscheck checkbounds(view.indices, idx)
    idxs = view.indices[idx]
    dims = map(length, idxs)
    output = similar(parent(view), dims...)
    @inbounds for idx in CartesianIndices(output)
        tidx, pidx = Tuple(idx)
        i = idxs[1][tidx]
        j = idxs[2][pidx]
        output[idx] = view.parent[i, j]
    end
    return output
end

function Base.copyto!(view::MultiAnnulusView, mats::Vararg{<:AbstractMatrix})

    inverse!(view, view.parent, mats)
    return view
end

function inverse!(view::MultiAnnulusView, out, mats::Vararg{<:AbstractMatrix})
    for (idxs, mat) in zip(view.indices, mats)
        @inbounds for idx in CartesianIndices(mat)
            tidx, pidx = Tuple(idx)
            i = idxs[1][tidx]
            j = idxs[2][pidx]
            out[i, j] = mat[idx]
        end
    end
    return out
end

inverse!(view::MultiAnnulusView, out, mats) = inverse!(view, out, mats...)
inverse(view::MultiAnnulusView, mats::Vararg{<:AbstractMatrix}) = inverse!(view, fill!(similar(parent(view)), view.fill), mats...)
inverse(view::MultiAnnulusView, mats) = inverse(view, mats...)
