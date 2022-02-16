var documenterSearchIndex = {"docs":
[{"location":"sdi/#Spectral-Processing","page":"Spectral Processing","title":"Spectral Processing","text":"","category":"section"},{"location":"sdi/","page":"Spectral Processing","title":"Spectral Processing","text":"The following functions are used specifically for processing spectral differential imaging (SDI) tensors/cubes.","category":"page"},{"location":"sdi/#Index","page":"Spectral Processing","title":"Index","text":"","category":"section"},{"location":"sdi/","page":"Spectral Processing","title":"Spectral Processing","text":"Pages = [\"sdi.md\"]","category":"page"},{"location":"sdi/#API/Reference","page":"Spectral Processing","title":"API/Reference","text":"","category":"section"},{"location":"sdi/","page":"Spectral Processing","title":"Spectral Processing","text":"invscale\ninvscale_and_collapse\nscale_list\nscale\nscale_and_stack","category":"page"},{"location":"sdi/#HCIToolbox.invscale","page":"Spectral Processing","title":"HCIToolbox.invscale","text":"invscale(cube::AbstractArray{T,3}, scales)\n\nLinearly contract each frame in cube by the corresponding scale in scales. Uses bilinear interpolaiton internally.\n\n\n\n\n\ninvscale(frame::AbstractMatrix, scale)\n\nLinearly contract frame with the ratio scale. Uses bilinear interpolation internally.\n\n\n\n\n\n","category":"function"},{"location":"sdi/#HCIToolbox.invscale_and_collapse","page":"Spectral Processing","title":"HCIToolbox.invscale_and_collapse","text":"invscale_and_collapse(stack_cube::AbstractArray{T,3}, scales; kwargs...)\n\nGiven an SDI tensor that has been stacked into a cube, invscales each spectral slice and combines with collapse. The output will be cropped to size.\n\n\n\n\n\n","category":"function"},{"location":"sdi/#HCIToolbox.scale_list","page":"Spectral Processing","title":"HCIToolbox.scale_list","text":"scale_list(wavelengths)\n\nReturns a list of scaling factors for aligning SDI tensors from a list of wavelengths.\n\nExamples\n\njulia> scale_list([0.5, 2, 4])\n3-element Vector{Float64}:\n 8.0\n 2.0\n 1.0\n\n\n\n\n\n","category":"function"},{"location":"sdi/#HCIToolbox.scale","page":"Spectral Processing","title":"HCIToolbox.scale","text":"scale(cube::AbstractArray{T,3}, scales)\n\nLinearly stretch each frame in cube by the corresponding scale in scales. Uses bilinear interpolaiton internally. The output size can be specified or else we will choose the smallest size that contains the largest stretch.\n\n\n\n\n\nscale(frame::AbstractMatrix, scale)\n\nLinearly stretch frame with the ratio scale. Uses bilinear interpolation internally.\n\n\n\n\n\n","category":"function"},{"location":"sdi/#HCIToolbox.scale_and_stack","page":"Spectral Processing","title":"HCIToolbox.scale_and_stack","text":"scale_and_stack(spcube::AbstractArray{T,4}, scales)\n\nGiven a 4-D spectral ADI (SDI) tensor this function scales each temporal slice according to scales and then concatenates into a cube with nλ * nf frames.\n\n\n\n\n\n","category":"function"},{"location":"api/#Index","page":"Index","title":"Index","text":"","category":"section"},{"location":"api/","page":"Index","title":"Index","text":"Here is a dump of all documented functions and types","category":"page"},{"location":"api/","page":"Index","title":"Index","text":"","category":"page"},{"location":"geometry/#Geometries","page":"Geometries","title":"Geometries","text":"","category":"section"},{"location":"geometry/","page":"Geometries","title":"Geometries","text":"A common way of analyzing HCI data is to look at specific spatial extents, such as annuli. These geometries can be thought of as spatially filtering the input data.","category":"page"},{"location":"geometry/","page":"Geometries","title":"Geometries","text":"For example, to create a geometry that is a concentric circle with radius r, we could filter a single frame like this","category":"page"},{"location":"geometry/","page":"Geometries","title":"Geometries","text":"frame = ones(101, 101)\nidxs = CartesianIndices(frame)\nradius = 10\ncenter = (51, 51)\n# only include indices that are within circle\nidxs_inside_circle = filter(idxs) do idx\n    translated = idx.I .- center\n    dist = sqrt(sum(abs2, translated))\n    return dist ≤ radius\nend","category":"page"},{"location":"geometry/","page":"Geometries","title":"Geometries","text":"using these filtered indices we can mask the data with something like","category":"page"},{"location":"geometry/","page":"Geometries","title":"Geometries","text":"masked = zero(frame)\nmasked[idxs_inside_circle] = frames[idxs_inside_circle]","category":"page"},{"location":"geometry/","page":"Geometries","title":"Geometries","text":"more useful, though, is filtering the data. If we think of the frame as a sample of pixels unrolled into a vector, we can filter that vector and only use the pixels that are within the mask.","category":"page"},{"location":"geometry/","page":"Geometries","title":"Geometries","text":"filtered = frame[idxs_inside_circle]","category":"page"},{"location":"geometry/","page":"Geometries","title":"Geometries","text":"This is very convenient for statistical algorithms wince we are filtering the data instead of just masking it, which greatly reduces the number of pixels. For example, the circle defined above only uses 4% of the data, so why waste time processing the rest?","category":"page"},{"location":"geometry/#Index","page":"Geometries","title":"Index","text":"","category":"section"},{"location":"geometry/","page":"Geometries","title":"Geometries","text":"Pages = [\"geometry.md\"]","category":"page"},{"location":"geometry/#API/Reference","page":"Geometries","title":"API/Reference","text":"","category":"section"},{"location":"geometry/","page":"Geometries","title":"Geometries","text":"AnnulusView\nMultiAnnulusView\neachannulus\ninverse\ninverse!\ncopyto!","category":"page"},{"location":"geometry/#HCIToolbox.AnnulusView","page":"Geometries","title":"HCIToolbox.AnnulusView","text":"AnnulusView(cube::AbstractArray{T,3};\n            inner=0, outer=first(size(parent))/2 + 0.5,\n            fill=0)\n\nCut out an annulus with inner radius inner and outer radius outer. Values that fall outside of this region will be replaced with fill. This does not copy any data, it is merely a view into the data.\n\n\n\n\n\n(::AnnulusView)(asview=false)\n\nReturn the pixels that fall within the annulus as a matrix. This matrix is equivalent to unrolling each frame and then spatially filtering the pixels outside the annulus. If asview is true, the returned values will be a view of the parent array instead of a copy.\n\nExamples\n\njulia> ann = AnnulusView(ones(101, 101, 10); inner=5, outer=20);\n\njulia> X = ann();\n\njulia> size(X)\n(1188, 10)\n\n\n\n\n\n","category":"type"},{"location":"geometry/#HCIToolbox.MultiAnnulusView","page":"Geometries","title":"HCIToolbox.MultiAnnulusView","text":"MultiAnnulusView(cube::AbstractArray{T,3} width, radii; fill=0)\n\nCreate multiple annuli at each radius in radii with width width. Values that fall outside of these regions will be replaced with fill. This does not copy any data, it is merely a view into the data.\n\n\n\n\n\nMultiAnnulusView(cube::AbstractArray{T,3}, width;\n                 inner=0, outer=first(size(parent))/2 + 0.5,\n                 fill=0)\n\nCreate multiple annuli between inner and outer with width spacing. Values that fall outside of these regions will be replaced with fill. This does not copy any data, it is merely a view into the data.\n\n\n\n\n\n(::MultiAnnulusView)(idx, asview=false)\n\nReturn the idxth annulus as a matrix. This is equivalent to unrolling the frame and filtering out pixels outside of the idxth annulus. If asview is true, the returned values will be a view of the parent array instead of a copy.\n\nExamples\n\njulia> ann = MultiAnnulusView(ones(101, 101, 10), 5; inner=5, outer=30);\n\njulia> X = ann(1);\n\njulia> size(X)\n(248, 10)\n\njulia> X2 = ann(2);\n\njulia> size(X2)\n(404, 10)\n\nSee also\n\neachannulus\n\n\n\n\n\n","category":"type"},{"location":"geometry/#HCIToolbox.eachannulus","page":"Geometries","title":"HCIToolbox.eachannulus","text":"eachannulus(::MultiAnnulusView, asview=false)\n\nCreate a generator for each annulus in the view. If asview is true, the annuli will be returned as a view into the parent array instead of a copy.\n\nExamples\n\njulia> ann = MultiAnnulusView(ones(101, 101, 10), 5; inner=5, outer=30);\n\njulia> [size(X) for X in eachannulus(ann)]\n5-element Vector{Tuple{Int64, Int64}}:\n (248, 10)\n (404, 10)\n (560, 10)\n (716, 10)\n (880, 10)\n\n\n\n\n\n","category":"function"},{"location":"geometry/#HCIToolbox.inverse","page":"Geometries","title":"HCIToolbox.inverse","text":"inverse(::AnnulusView, mat::AbstractMatrix)\n\nGenerate a cube similar to the view with the pixels from mat. mat should have the same size as the matrix output from AnnulusView\n\nExamples\n\njulia> ann = AnnulusView(ones(101, 101, 10); inner=5, outer=20);\n\njulia> X = ann();\n\njulia> out = inverse(ann, -X);\n\njulia> out ≈ -ann\ntrue\n\n\n\n\n\ninverse(::MultiAnnulusView, idx, mat)\ninverse(::MultiAnnulusView, mats...)\n\nGenerate a cube similar to the view using the given pixel matrices. The pixels from mat will be put into the location of the idxth annulus. mat should have the same size as the output matrices generated by MultiAnnulusView. If multiple matrices are supplied, it is assumed each one corresponds to each annulus in the view.\n\nExamples\n\nExpand a single annulus-\n\njulia> ann = MultiAnnulusView(ones(101, 101, 10), 5; inner=5, outer=30);\n\njulia> X = ann(1);\n\njulia> out = inverse(ann, 1, -X);\n\njulia> sum(out) == -sum(X)\ntrue\n\nexpand many annuli-\n\njulia> Xs = [-X for X in eachannulus(ann)];\n\njulia> out = inverse(ann, Xs);\n\njulia> out ≈ -ann\ntrue\n\n\n\n\n\n","category":"function"},{"location":"geometry/#HCIToolbox.inverse!","page":"Geometries","title":"HCIToolbox.inverse!","text":"inverse!(::AnnulusView, out, mat)\n\nIn-place version of inverse that fills out in-place.\n\n\n\n\n\ninverse!(::MultiAnnulusView, out, idx, mat)\ninverse!(::MultiAnnulusView, out, mats...)\n\nIn-place version of inverse that fills out with annuli defined by the geometry of the view.\n\n\n\n\n\n","category":"function"},{"location":"geometry/#Base.copyto!","page":"Geometries","title":"Base.copyto!","text":"copyto!(::AnnulusView, mat::AbstractMatrix)\n\nCopy the pixels from mat into the pixels in the annulus. mat should have the same size as the matrix output from AnnulusView\n\nExamples\n\njulia> ann = AnnulusView(ones(101, 101, 10); inner=5, outer=20);\n\njulia> X = ann();\n\njulia> new_ann = copyto!(ann, -X);\n\njulia> new_ann() ≈ -X\ntrue\n\n\n\n\n\ncopyto!(::MultiAnnulusView, idx, mat)\ncopyto!(::MultiAnnulusView, mats...)\n\nCopy the pixels from mat into the pixels in the idxth annulus. mat should have the same size as the matrices generated by MultiAnnulusView. If multiple matrices are supplied, it is assumed each one corresponds to each annulus in the view.\n\nExamples\n\nUpdate a single annulus-\n\njulia> ann = MultiAnnulusView(ones(101, 101, 10), 5; inner=5, outer=30);\n\njulia> X = ann(1);\n\njulia> new_ann = copyto!(ann, 1, -X);\n\njulia> new_ann(1) ≈ -X\ntrue\n\nupdate each annulus-\n\njulia> ann = MultiAnnulusView(ones(101, 101, 10), 5; inner=5, outer=30);\n\njulia> Xs = [-X for X in eachannulus(ann)];\n\njulia> new_ann = copyto!(copy(ann), Xs);\n\njulia> new_ann ≈ -ann\ntrue\n\n\n\n\n\n","category":"function"},{"location":"processing/#Processing","page":"Processing","title":"Processing","text":"","category":"section"},{"location":"processing/","page":"Processing","title":"Processing","text":"There are a variety of functions defined by HCIToolbox to process HCI data. In particular, routines like derotating cubes, collapsing data cubes, scaling SDI data, etc. are available.","category":"page"},{"location":"processing/","page":"Processing","title":"Processing","text":"tip: Multi-threading\nMany of the methods that work on cubes of data try to multi-thread the operations along the time axis. Make sure you set the environment variable JULIA_NUM_THREADS before staring your runtime to take advantage of this.","category":"page"},{"location":"processing/#Index","page":"Processing","title":"Index","text":"","category":"section"},{"location":"processing/","page":"Processing","title":"Processing","text":"Pages = [\"processing.md\"]","category":"page"},{"location":"processing/#API/Reference","page":"Processing","title":"API/Reference","text":"","category":"section"},{"location":"processing/","page":"Processing","title":"Processing","text":"collapse\ncollapse!\ncrop\ncropview\nderotate\nderotate!\nexpand\nflatten\nshift_frame\nshift_frame!","category":"page"},{"location":"processing/#HCIToolbox.collapse","page":"Processing","title":"HCIToolbox.collapse","text":"collapse(cube; method=median, fill=0, degree=Linear())\ncollapse(cube, angles; method=:deweight, fill=0, degree=Linear())\n\nCombine all the frames of a cube using method. If angles are provided, will use derotate before combining.\n\nIf method is :deweight, the method of Bottom et al. 2017 will be used in which the combined image will be the derotated weighted sum of the frames weighted by the temporal variance. Keyword arguments will be passed to derotate.\n\nReferences\n\nBottom et al. 2017 \"Noise-weighted Angular Differential Imaging\"\n\nExamples\n\njulia> X = ones(3, 3, 2);\n\njulia> collapse(X)\n3×3 Matrix{Float64}:\n 1.0  1.0  1.0\n 1.0  1.0  1.0\n 1.0  1.0  1.0\n\njulia> collapse(X, [0, 90])\n3×3 Matrix{Float64}:\n 1.0  1.0  0.5\n 1.0  1.0  1.0\n 0.5  0.5  0.5\n\njulia> collapse(X, [0, 90], fill=NaN)\n3×3 Matrix{Float64}:\n   1.0    1.0  NaN\n   1.0    1.0    1.0\n NaN    NaN    NaN\n\n\nSee Also\n\ncollapse!\n\n\n\n\n\n","category":"function"},{"location":"processing/#HCIToolbox.collapse!","page":"Processing","title":"HCIToolbox.collapse!","text":"collapse!(cube, angles; method=:deweight, fill=0)\n\nAn in-place version of the derotating collapse. The only difference is in this version the cube will be derotated in-place.\n\n\n\n\n\n","category":"function"},{"location":"processing/#HCIToolbox.crop","page":"Processing","title":"HCIToolbox.crop","text":"crop(frame, size; center=center(frame), force=false)\ncrop(cube, size; center=center(cube), force=false)\n\nCrop a frame or cube to size. size can be a tuple or an integer, which will make a square crop. The indices will be relative to center. To avoid uneven (odd) cropping, we may change size. To disable this behavior, set force to true. To avoid allocations, consider cropview.\n\n\n\n\n\n","category":"function"},{"location":"processing/#HCIToolbox.cropview","page":"Processing","title":"HCIToolbox.cropview","text":"cropview(cube::AbstractArray{T, 3}, size; center=center(cube), force=false)\n\nCrop a frame to size, returning a view of the frame. size can be a tuple or an integer, which will make a square crop. The indices will be relative to center. To avoid uneven (odd) cropping, we may change size. To disable this behavior, set force to true.\n\nSee Also\n\ncrop\n\n\n\n\n\ncropview(frame::AbstractMatrix, size; center=center(frame), force=false)\n\nCrop a frame to size, returning a view of the frame. size can be a tuple or an integer, which will make a square crop. The indices will be relative to center. To avoid uneven (odd) cropping, we may change size. To disable this behavior, set force to true.\n\nSee Also\n\ncrop\n\n\n\n\n\n","category":"function"},{"location":"processing/#HCIToolbox.derotate","page":"Processing","title":"HCIToolbox.derotate","text":"derotate(frame, angle; fill=0, degree=Linear())\n\nRotates frame counter-clockwise by angle, given in degrees. This is merely a convenient wrapper around ImageTransformations.imrotate.\n\n\n\n\n\nderotate(cube, angles; fill=0, degree=Linear())\n\nRotates an array using the given angles in degrees.\n\nThis will rotate frame i counter-clockwise. Any values outside the original axes will be replaced with fill. If the given angles are true parallactic angles, the resultant cube will have each frame aligned with top pointing North. degree is the corresponding Interpolations.Degree for the B-Spline used to subsample the pixel values.\n\nExamples\n\njulia> X = zeros(3, 3, 1); X[2, 1, 1] = 1;\n\njulia> X[:, :, 1]\n3×3 Matrix{Float64}:\n 0.0  0.0  0.0\n 1.0  0.0  0.0\n 0.0  0.0  0.0\n\njulia> derotate(X, [90])[:, :, 1]\n3×3 Matrix{Float64}:\n 0.0  0.0          0.0\n 0.0  4.44089e-16  0.0\n 0.0  0.0          0.0\n\nSee Also\n\nderotate!\n\n\n\n\n\n","category":"function"},{"location":"processing/#HCIToolbox.derotate!","page":"Processing","title":"HCIToolbox.derotate!","text":"derotate!(cube, angles; fill=0, degree=Linear())\n\nIn-place version of derotate which modifies cube.\n\n\n\n\n\n","category":"function"},{"location":"processing/#HCIToolbox.expand","page":"Processing","title":"HCIToolbox.expand","text":"expand(matrix)\n\nGiven a matrix of size (z, n), returns a cube of size (x, x, n) where x=√z.\n\nWill throw an error if z is not a perfect square.\n\nExamples\n\njulia> X = ones(4, 3)\n4×3 Matrix{Float64}:\n 1.0  1.0  1.0\n 1.0  1.0  1.0\n 1.0  1.0  1.0\n 1.0  1.0  1.0\n\njulia> expand(X)[:, :, 1]\n2×2 Matrix{Float64}:\n 1.0  1.0\n 1.0  1.0\n\nSee Also\n\nflatten\n\n\n\n\n\n","category":"function"},{"location":"processing/#HCIToolbox.flatten","page":"Processing","title":"HCIToolbox.flatten","text":"flatten(cube)\n\nGiven a cube of size (x, y, n) returns a matrix with size (x * y, n) where each row is a flattened image from the cube.\n\nExamples\n\njulia> X = ones(2, 2, 3);\n\njulia> flatten(X)\n4×3 Matrix{Float64}:\n 1.0  1.0  1.0\n 1.0  1.0  1.0\n 1.0  1.0  1.0\n 1.0  1.0  1.0\n\nSee Also\n\nexpand\n\n\n\n\n\n","category":"function"},{"location":"processing/#HCIToolbox.shift_frame","page":"Processing","title":"HCIToolbox.shift_frame","text":"shift_frame(frame, dx, dy; fill=0)\nshift_frame(frame, dpos; fill=0)\n\nShifts frame by dx and dy with bilinear interpolation. If necessary, empty indices will be filled with fill.\n\nExamples\n\njulia> shift_frame([0 0 0; 0 1 0; 0 0 0], 1, -1)\n3×3 Matrix{Float64}:\n 0.0  0.0  0.0\n 0.0  0.0  0.0\n 1.0  0.0  0.0\n\njulia> shift_frame(ans, (-1, 1), fill=NaN)\n3×3 Matrix{Float64}:\n NaN    0.0    0.0\n NaN    1.0    0.0\n NaN  NaN    NaN\n\n\n\n\n\nshift_frame(cube, dx, dy; fill=0)\nshift_frame(cube, dpos; fill=0)\n\nShift each frame of cube by dx and dy, which can be integers or vectors. The change in position can be given as a tuple, which can also be put into a vector to use across the cube. If a frame is shifted outside its axes, the empty indices will be filled with fill.\n\nSee Also\n\nshift_frame!\n\n\n\n\n\n","category":"function"},{"location":"processing/#HCIToolbox.shift_frame!","page":"Processing","title":"HCIToolbox.shift_frame!","text":"shift_frame!(cube, dx, dy; fill=0)\nshift_frame!(cube, dpos; fill=0)\n\nIn-place version of shift_frame which modifies cube.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = HCIToolbox","category":"page"},{"location":"#HCIToolbox.jl","page":"Home","title":"HCIToolbox.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"(Image: GitHub) (Image: Build Status) (Image: PkgEval) (Image: Coverage) (Image: License)","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"From Julia enter Pkg mode","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia>]\n\n(1.3) pkg> add HCIToolbox","category":"page"},{"location":"#License","page":"Home","title":"License","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This work is distributed under the MIT \"expat\" license. See LICENSE for more information.","category":"page"},{"location":"utils/#Utilities","page":"Utilites","title":"Utilities","text":"","category":"section"},{"location":"utils/","page":"Utilites","title":"Utilites","text":"The following routines are general utility or otherwise don't fit into a category","category":"page"},{"location":"utils/#Index","page":"Utilites","title":"Index","text":"","category":"section"},{"location":"utils/","page":"Utilites","title":"Utilites","text":"Pages = [\"utils.md\"]","category":"page"},{"location":"utils/#API/Reference","page":"Utilites","title":"API/Reference","text":"","category":"section"},{"location":"utils/","page":"Utilites","title":"Utilites","text":"mask_circle\nmask_circle!\nmask_annulus\nmask_annulus!\nnormalize_par_angles\nnormalize_par_angles!\nradial_profile","category":"page"},{"location":"utils/#HCIToolbox.mask_circle","page":"Utilites","title":"HCIToolbox.mask_circle","text":"mask_circle(::AbstractMatrix, npix; fill=0, center=center(arr))\nmask_circle(::AbstractArray, npix; fill=0, center=center(arr))\n\nMask the inner-circle of an image with radius npix with value fill. Note that the input type must be compatible with the fill value's type. If the input is a cube it will mask each frame individually.\n\nSee Also\n\nmask_circle!\n\n\n\n\n\n","category":"function"},{"location":"utils/#HCIToolbox.mask_circle!","page":"Utilites","title":"HCIToolbox.mask_circle!","text":"mask_circle!(::AbstractMatrix, npix; fill=0, center=center(arr))\nmask_circle!(::AbstractArray, npix; fill=0, center=center(arr))\n\nIn-place version of mask_circle\n\n\n\n\n\n","category":"function"},{"location":"utils/#HCIToolbox.mask_annulus","page":"Utilites","title":"HCIToolbox.mask_annulus","text":"mask_annulus(::AbstractMatrix, npix_in, npix_out; fill=0, center=center(arr))\n\nMask an annular region of an image with inner-radius npix_in, outer-radius npix_out with value fill. Note that the input type must be compatible with the fill value's type.\n\nSee Also\n\nmask_annulus!\n\n\n\n\n\n","category":"function"},{"location":"utils/#HCIToolbox.mask_annulus!","page":"Utilites","title":"HCIToolbox.mask_annulus!","text":"mask_annulus!(::AbstractMatrix, npix_in, npix_out; fill=0, center=center(arr))\n\nIn-place version of mask_annulus\n\n\n\n\n\n","category":"function"},{"location":"utils/#HCIToolbox.normalize_par_angles","page":"Utilites","title":"HCIToolbox.normalize_par_angles","text":"normalize_par_angles(angles)\n\nEnsures parallactic angle list (in degrees) is positive monotonic with no jumps greater than 180°.\n\nExamples\n\njulia> normalize_par_angles([-10, 20, 190])\n3-element Vector{Int64}:\n 350\n  20\n 190\n\n\n\n\n\n","category":"function"},{"location":"utils/#HCIToolbox.normalize_par_angles!","page":"Utilites","title":"HCIToolbox.normalize_par_angles!","text":"normalize_par_angles!(angles)\n\nIn-place version of normalize_par_angles\n\n\n\n\n\n","category":"function"},{"location":"utils/#HCIToolbox.radial_profile","page":"Utilites","title":"HCIToolbox.radial_profile","text":"radial_profile(image; center=center(image))\n\nCalculates the radial profile of image, centered around center. Returns the radii and corresponding profile.\n\nnote: Understanding bin edges\nThe radii returned here are integral, for example [0, 1, 2, ...]. These are equivalent to the centers of the contours of an annulus between r - 0.5, r + 0.5 pixels. These annuli are centered on pixels (at least those orthogonal to the axes). \n\nExamples\n\njulia> X = [0 1 1 1 0\n            1 2 2 2 1\n            1 2 3 2 1\n            1 2 2 2 1\n            0 1 1 1 0];\n\njulia> r, prof = radial_profile(X);\n\njulia> r .=> prof # radius => value\n4-element Vector{Pair{Int64, Float64}}:\n 0 => 3.0\n 1 => 2.0\n 2 => 1.0\n 3 => 0.0\n\njulia> radial_profile([1 2; 2 2], center=(1, 1))\n([0, 1], [1.0, 2.0])\n\n\n\n\n\n","category":"function"},{"location":"inject/#Signal-Injection","page":"Signal Injection","title":"Signal Injection","text":"","category":"section"},{"location":"inject/","page":"Signal Injection","title":"Signal Injection","text":"The following functions are used for injecting fake signal into HCI data.","category":"page"},{"location":"inject/#Index","page":"Signal Injection","title":"Index","text":"","category":"section"},{"location":"inject/","page":"Signal Injection","title":"Signal Injection","text":"Pages = [\"inject.md\"]","category":"page"},{"location":"inject/#API/Reference","page":"Signal Injection","title":"API/Reference","text":"","category":"section"},{"location":"inject/","page":"Signal Injection","title":"Signal Injection","text":"inject\ninject!","category":"page"},{"location":"inject/#HCIToolbox.inject","page":"Signal Injection","title":"HCIToolbox.inject","text":"inject(frame::AbstractMatrix, psf, [angle]; x, y, amp=1, center=center(frame), kwargs...)\n\nInjects the psf into frame at the given position.\n\nExamples\n\ninject works with two types of PSF models: matrices and synthetic models. \n\nMatrices\n\nIf you pass an AbstractMatrix to inject, you can optionally specify degree and fill as keyword arguments. By default, degree=Interpolations.Linear() and fill=0. These are used to create an Interpolations.AbstractExtrapolation type which can be arbitrarily transformed. From here, the x, y, and amp arguments will determine the position and will be optionally rotated angle degrees counter-clockwise around the center.\n\njulia> inject(zeros(5, 5), ones(1, 1); x=4, y=3, amp=2)\n5×5 Matrix{Float64}:\n 0.0  0.0  0.0  0.0  0.0\n 0.0  0.0  0.0  0.0  0.0\n 0.0  0.0  0.0  0.0  0.0\n 0.0  0.0  2.0  0.0  0.0\n 0.0  0.0  0.0  0.0  0.0\n\njulia> inject(zeros(5, 5), ones(3, 3), 90; x=4, y=3, amp=2)\n5×5 Matrix{Float64}:\n 0.0  0.0  0.0  0.0  0.0\n 0.0  0.0  2.0  2.0  2.0\n 0.0  0.0  2.0  2.0  2.0\n 0.0  0.0  2.0  2.0  2.0\n 0.0  0.0  0.0  0.0  0.0\n\nSynthetic Models\n\nThe synthetic models from PSFModels.jl can easily be used, too. In this case, any additional keyword arguments are directly passed to the given model-\n\njulia> using PSFModels\n\njulia> inject(zeros(5, 5), gaussian; x=4, y=3, amp=2, fwhm=2)\n5×5 Matrix{Float64}:\n 0.000244141  0.00195312  0.00390625  0.00195312  0.000244141\n 0.0078125    0.0625      0.125       0.0625      0.0078125\n 0.0625       0.5         1.0         0.5         0.0625\n 0.125        1.0         2.0         1.0         0.125\n 0.0625       0.5         1.0         0.5         0.0625\n\njulia> inject(zeros(5, 5), airydisk, 45; x=4, y=3, amp=2, fwhm=2, ratio=0.3)\n5×5 Matrix{Float64}:\n 0.042946   0.0944566   0.0741111  0.06456   0.0914308\n 0.0944566  0.0191043   0.054543   0.108174  0.00031632\n 0.0741111  0.054543    0.929442   1.29573   0.307196\n 0.06456    0.108174    1.29573    1.76798   0.470669\n 0.0914308  0.00031632  0.307196   0.470669  0.0620712\n\n\n\n\n\ninject(cube, psf, [angles]; x, y, amp=1, kwargs...)\n\nInjects the matrix psf into each frame of cube at the given position.\n\nIf angles are provided, each position will be rotated by the given parallactic angles in degrees, centered at center, which is the center of each frame, by default.\n\n\n\n\n\n","category":"function"},{"location":"inject/#HCIToolbox.inject!","page":"Signal Injection","title":"HCIToolbox.inject!","text":"inject!(cube, psf, [angles]; kwargs...)\n\nIn-place version of inject which modifies cube.\n\n\n\n\n\n","category":"function"}]
}
