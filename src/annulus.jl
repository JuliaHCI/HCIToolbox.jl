using ImageTransformations: center

function get_annulus_segments(data, inner_radius, width; theta_init=0, optim_scale_factor=1, mode=:index)
    cy, cx = center(data)
    ys, xs = axes(data)
    dist = @. sqrt((xs' - cx)^2 + (ys - cy)^2)
    ϕ = @. atan(ys - cy, xs - cx)
    ϕ_rot = mod2pi.(ϕ)
    outer_radius = inner_radius + optim_scale_factor * width

    ϕ_start = deg2rad(theta_init)
    ϕ_end = ϕ_start + 2π

    if ϕ_start < 2π && ϕ_end > 2π
        mask = @. (dist ≥ inner_radius) & (dist < outer_radius) & (ϕ_rot ≥ ϕ_start) & (ϕ_rot ≤ 2π) | (dist ≥ inner_radius) & (dist < outer_radius) & (ϕ_rot ≥ 0) & (ϕ_rot < ϕ_end - 2π)
    elseif ϕ_start ≥ 2π && ϕ_end > 2π
        mask = @. (dist ≥ inner_radius) & (dist < outer_radius) & (ϕ_rot ≥ ϕ_start - 2π) & (ϕ_rot < ϕ_end - 2π)
    else
        mask = @. (dist ≥ inner_radius) & (dist < outer_radius) & (ϕ_rot ≥ ϕ_start) & (ϕ_rot < ϕ_end)
    end

    return _convert_mask(Val(mode), mask, data)
end

function get_annulus_segments(data, inner_radius, width, nsegments; theta_init=0, optim_scale_factor=1, mode=:index)
    cy, cx = center(data)
    ys, xs = axes(data)
    dist = @. sqrt((xs' - cx)^2 + (ys - cy)^2)
    ϕ_stride = deg2rad(ceil(Int, 360 / nsegments))
    ϕ = @. atan(ys - cy, xs - cx)
    ϕ_rot = mod2pi.(ϕ)
    outer_radius = inner_radius + optim_scale_factor * width

    masks = similar(data, BitArray{2}, nsegments)
    for i in 1:nsegments
        ϕ_start = deg2rad(theta_init) + i * ϕ_stride
        ϕ_end = ϕ_start + ϕ_stride

        if ϕ_start < 2π && ϕ_end > 2π
            mask = @. (dist ≥ inner_radius) & (dist < outer_radius) & (ϕ_rot ≥ ϕ_start) & (ϕ_rot ≤ 2π) | (dist ≥ inner_radius) & (dist < outer_radius) & (ϕ_rot ≥ 0) & (ϕ_rot < ϕ_end - 2π)
        elseif ϕ_start ≥ 2π && ϕ_end > 2π
            mask = @. (dist ≥ inner_radius) & (dist < outer_radius) & (ϕ_rot ≥ ϕ_start - 2π) & (ϕ_rot < ϕ_end - 2π)
        else
            mask = @. (dist ≥ inner_radius) & (dist < outer_radius) & (ϕ_rot ≥ ϕ_start) & (ϕ_rot < ϕ_end)
        end
        @inbounds masks[i] = mask
    end

    return _convert_mask.(Val(mode), masks, Ref(data))
end

_convert_mask(::Val{:index}, mask, data) = mask
_convert_mask(::Val{:value}, mask, data) = data[mask]
_convert_mask(::Val{:mask}, mask, data) = data .* mask
