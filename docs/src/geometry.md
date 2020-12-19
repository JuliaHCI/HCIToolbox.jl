# Geometries

A common way of analyzing HCI data is to look at specific spatial extents, such as annuli. These geometries can be thought of as *spatially filtering* the input data.

For example, to create a geometry that is a concentric circle with radius `r`, we could filter a single frame like this

```julia
frame = ones(101, 101)
idxs = CartesianIndices(frame)
radius = 10
center = (51, 51)
# only include indices that are within circle
idxs_inside_circle = filter(idxs) do idx
    translated = idx.I .- center
    dist = sqrt(sum(abs2, translated))
    return dist ≤ radius
end
```

using these filtered indices we can *mask* the data with something like

```julia
masked = zero(frame)
masked[idxs_inside_circle] = frames[idxs_inside_circle]
```

more useful, though, is filtering the data. If we think of the frame as a sample of pixels unrolled into a vector, we can filter that vector and only use the pixels that are within the mask.

```julia
filtered = frame[idxs_inside_circle]
```

This is very convenient for statistical algorithms wince we are *filtering* the data instead of just *masking* it, which greatly reduces the number of pixels. For example, the circle defined above only uses 4% of the data, so why waste time processing the rest?

## Index

```@index
Pages = ["geometry.md"]
```

## API/Reference

```@docs
AnnulusView
MultiAnnulusView
eachannulus
inverse
inverse!
copyto!
```
