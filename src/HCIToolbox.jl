module HCIToolbox

export derotate,
       derotate!,
       expand,
       flatten,
       collapse,
       collapse!,
       crop,
       cropview,
       mask_circle!,
       mask_circle,
       mask_annulus!,
       mask_annulus,
       normalize_par_angles, 
       normalize_par_angles!,
       inject,
       inject!,
       scale,
       invscale,
       shift_frame,
       shift_frame!,
       scale_and_stack,
       invscale_and_collapse,
       scale_list
       # geometry
       AnnulusView
       inverse!
       inverse

# Utilities for dealing with cubes like derotating and median combining
include("morphology.jl")
include("inject.jl")

# Utilities for mask design
include("masking.jl")
include("angles.jl")

include("scaling.jl")

include("geometry.jl")

end
