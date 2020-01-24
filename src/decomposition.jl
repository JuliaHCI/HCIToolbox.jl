using MultivariateStats
using Statistics

const mvs = MultivariateStats


abstract type Design end
struct PCA <: Design 
    
end
struct NMF <: Design end

struct Pairet{T <: Design} <: Design
    strategy::T
end


function reduce(::Type{Pairet{<:PCA}}, cube::DataCube; pca_kwargs...)
    S = []
    reduced = []

    X = Matrix(cube)

    initial_pca = mvs.fit(mvs.PCA, X; maxoutdim = 1)
    

end


