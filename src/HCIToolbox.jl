module HCIToolbox

export design,
       PCA,
       NMF,
       Pairet,
       Median,
       Mean,
       rotate,
       rotate!,
       derotate,
       derotate!,
       flatten,
       expand,
       combine,
       mask_inner!,
       mask_inner
    

# Utilities for dealing with cubes like derotating and median combining
include("stacking.jl")

# The core decomposition routines
include("decomposition.jl")

# others
include("masking.jl")

end
