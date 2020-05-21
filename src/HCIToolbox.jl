module HCIToolbox

export derotate,
       derotate!,
       expand,
       flatten,
       collapse,
       collapse!,
       mask_circle!,
       mask_circle,
       mask_annulus!,
       mask_annulus,
       normalize_par_angles, 
       normalize_par_angles!,
       inject,
       inject!,
       shift_frame,
       shift_frame!

# Utilities for dealing with cubes like derotating and median combining
include("morphology.jl")
include("inject.jl")

# Utilities for mask design
include("masking.jl")
include("angles.jl")

end
