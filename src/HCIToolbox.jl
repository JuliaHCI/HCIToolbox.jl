module HCIToolbox

export rotate,
       rotate!,
       derotate,
       derotate!,
       flatten,
       expand,
       combine,
       mask_circle!,
       mask_circle
    

# Utilities for dealing with cubes like derotating and median combining
include("morphology.jl")

# Utilities for mask design
include("masking.jl")

end
