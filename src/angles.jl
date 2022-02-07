"""
    normalize_par_angles!(angles)

In-place version of [`normalize_par_angles`](@ref)
"""
function normalize_par_angles!(angles)
    # convert to positive
    @. angles[angles < 0] += 360
    idxs = sortperm(angles)


    correct = findfirst(d -> abs(d) > 180, diff(angles[idxs]))
    if !isnothing(correct)
        @. angles[angles < 180] += 360
    end

    return angles
end

"""
    normalize_par_angles(angles)

Ensures parallactic angle list (in degrees) is positive monotonic with no jumps greater than 180°.

# Examples
```jldoctest
julia> normalize_par_angles([-10, 20, 190])
3-element Vector{Int64}:
 350
  20
 190
```
"""
normalize_par_angles(angles) = normalize_par_angles!(deepcopy(angles))
