module Multigrid
using OffsetArrays
using LinearAlgebra
using SparseArrays
using Parameters

export MGConfig
export LeftBnd, RightBnd, TopBnd, BottomBnd, FrontBnd, BackBnd
export Dirichlet, Neumann
export CartesianConnector, CylindricalConnector

include("redblack.jl")
include("boundaries.jl")

    
""" A connector is a type that allows us for example to implement
cylindrical symmetry in a generic code without a performance overhead.

The connector must be able to compute from a given grid coordinate and
a stencil shift a transformation of the laplacian term (generally a 
multiplication).
"""
abstract type AbstractConnector end

struct CartesianConnector <: AbstractConnector end
(::CartesianConnector)(::CartesianIndex, ::CartesianIndex, x) = x

# For performance we store the cylindrical dimension in a type parameters
struct CylindricalConnector{D} <: AbstractConnector end


@with_kw struct MGConfig{T, TBC<:Tuple, C<:AbstractConnector}
    " Boundary condition as a tuple of tuples (boundary, condition)."
    bc::TBC

    " Geometrical connector to specify e.g. cylindrical symmetry. "
    conn::C
    
    " Multiplication constant for the lhs. "
    s::T

    " Number of levels of coarsening. "
    levels::Int

    " Allowed tolerance. "
    tolerance::T
    
    " Smoothing iterations before restriction/interpolation"
    smooth1::Int

    " Smoothing iterations after restriction/interpolation"
    smooth2::Int
end


"""
Pre-allocated space and matrix factorization struct.
"""
struct Workspace{T, TA <: AbstractArray{T}, M}
    res::Vector{TA}
    res1::Vector{TA}
    sol::Vector{TA}

    btop::Vector{T}
    utop::Vector{T}
    
    mat::M
end



function (::CylindricalConnector{D})(c::CartesianIndex, d::CartesianIndex, x) where D
    x * (1.0 + d[D] / (2 * c[D] - 1))
end

function inranges(a::AbstractArray{T, N}) where {T, N}
    ng = 1 .- first.(axes(a))
    l = last.(axes(a)) .- ng
    ntuple(n -> 1:l[n], Val(N))
end

innerindices(a) = CartesianIndices(inranges(a))
innersize(a) = length.(inranges(a))

function simcoarser(a)
    s = innersize(a)
    zeros(map(n->0:(n ÷ 2 + 1), s)...)    
end


function simfiner(a)
    s = innersize(a)
    zeros(map(n->0:(n * 2 + 1), s)...)    
end


@inline function lplstencil(a::AbstractArray{T, N}) where {T, N}
    z = zero(CartesianIndex{N})
    pos = ntuple(i->Base.setindex(z,  1, i), Val(N))
    neg = ntuple(i->Base.setindex(z, -1, i), Val(N))

    (pos..., neg...)
end


@inline function cubestencil(a::AbstractArray{T, N}) where {T, N}    
    CartesianIndex.(__unitvert(size(a)))
end


__unitvert(::Tuple{}) = ((),)


function __unitvert(tpl)
    pre = __unitvert(Base.tail(tpl))
    (map(x -> (0, x...), pre)..., map(x -> (1, x...), pre)...) 
end


function binterpweights(st)
    map(s -> prod(map(si -> 3 - 2si, Tuple(s))) / 4^length(s), st)
end


@inline function laplacian(u, ind, st, c::AbstractConnector)
    @inbounds s = -length(st) * u[ind]
    for j in st
        @inbounds s += c(ind, j, u[ind + j])
    end
    s
end


#@inline laplacian(u, ind, st) = laplacian(u, ind, st, CartesianConnector())

    
"""
   Update the potential `u` with Gauss-Seidel using the source `b`.
   `ω` is an over-relaxation parameter.
"""
function gauss_seidel!(u, b, ω, c::AbstractConnector)
    @assert size(u) == size(b)
    st = lplstencil(u)

    for parity in (false, true)
        redblack(u, parity) do ind
            l = laplacian(u, ind, st, c)
            @inbounds u[ind] += ω * (l + b[ind]) / length(st)
        end
    end
end
gauss_seidel!(u, b, ω) = gauss_seidel!(u, b, ω, CartesianConnector())


"""
   Computes the residual of Laplace operator acting on u with rhs -b.
   The function computes Lu + s b where L is the discrete laplace operator.
"""
function residual!(r, u, b, s, c::AbstractConnector)
    st = lplstencil(u)

    #Threads.@threads
    for ind in innerindices(u)
        l = laplacian(u, ind, st, c)
        @inbounds r[ind] = s * b[ind] + l
    end
end

residual!(r, u, b, c::AbstractConnector) = residual!(r, u, b, 1.0, c)
residual!(r, u, b, s) = residual!(r, u, b, s, CartesianConnector())


function residualnorm(u, b, c::AbstractConnector)
    r = zeros(axes(u))
    residual!(r, u, b, c)
    norm(r)
end
residualnorm(u, b) = residualnorm(u, b, CartesianConnector())



"""
    Restrict coarse grid `rh` into `r`.
"""
function restrict!(rh, r)
    st = cubestencil(r)
    
    Threads.@threads for irh in innerindices(rh)
        ir = 2 * (irh - oneunit(irh)) + oneunit(irh)
        s = zero(eltype(r))

        for j in st
            @inbounds s += r[ir + j]
        end

        @inbounds rh[irh] = 4 * s / length(st)
    end
end


"""
    Interpolate form `rh` into `r`, adding it to the value already stored there.
"""
function interpolate!(r, rh, update::Type{Val{V}}=Val{false}) where {V}
    st = cubestencil(r)
    weights = binterpweights(st)
    
    Threads.@threads for irh in innerindices(rh)
        ir = 2 * (irh - oneunit(irh)) + oneunit(irh)
        for s in st
            # We will compute the value of cell in F at this location
            indf = ir + s

            # Delta to the furthest cell in H that plays into the interpolation
            δ = 2 * s - oneunit(irh)

            # Now we run over cells in H to compute the interpolation
            s = zero(eltype(r))
            for (w, sh) in zip(weights, st)
                @inbounds s += w * rh[irh + CartesianIndex(Tuple(δ) .* Tuple(sh))]
            end
            if V
                r[indf] += s
            else
                r[indf] = s
            end
        end
    end
end


function buildmat(x, bc)
    rngs = inranges(x)
    cinds = innerindices(x)
    lin = LinearIndices(cinds)
    
    n = length(cinds)
    mat = spzeros(eltype(x), n, n)
    st = lplstencil(x)
    
    for ind in cinds
        mat[lin[ind], lin[ind]] -= length(st)
        for d in st
            newind = ind + d
            m = 1.0
            if !(newind in cinds)
                for (b, c) in bc
                    app, ind2, m2 = matcoef(rngs, newind, b, c)
                    if app
                        newind = ind2
                        m = m2
                        break
                    end
                end
            end
            if newind in cinds
                mat[lin[ind], lin[newind]] += m
            end
        end
    end
    factorize(mat)
end


function allocate(conf, x)
    @unpack levels, bc = conf
    
    res = typeof(x)[]
    sol = typeof(x)[]
    res1 = typeof(x)[]
    
    push!(res, zeros(axes(x)))
    push!(sol, zeros(axes(x)))
    push!(res1, zeros(axes(x)))

    for i in 1:levels
        push!(res, simcoarser(last(res)))
        push!(res1, simcoarser(last(res1)))
        push!(sol, simcoarser(last(sol)))
    end

    btop = vec(zeros(innersize(sol[end])...))
    utop = vec(zeros(innersize(sol[end])...))
    
    mat = buildmat(sol[end], bc)
    Workspace(res, res1, sol, btop, utop, mat)
end



""" 

Multigrid V-cycle. Improves a guess for the discrete 3D Poisson 
equation A.x == -b  where A is the discrete Laplacian operator with h=1.
    
"""
function mgv!(conf::MGConfig, x, b, level, ws)
    @assert size(x) == size(b)
    @unpack bc, conn, smooth1, smooth2, levels = conf
    
    st = lplstencil(x)
    
    if level == levels
        for i in 1:50
            apply!(x, bc)
            gauss_seidel!(x, b, 1.0, conn)
        end
        return x
    end

    for i in 1:smooth1
        apply!(x, bc)
        gauss_seidel!(x, b, 1.0, conn)
    end

    # We need extra space here to avoid clashing with the residuals
    # computed in fmg!
    r = ws.res1[level + 1]
    
    apply!(x, bc)
    residual!(r, x, b, conn)
    
    rh = ws.res[level + 2]
    
    apply!(r, bc)
    restrict!(rh, r)

    xh = ws.sol[level + 2]
    xh .= 0
    
    mgv!(conf, xh, rh, level + 1, ws)
    interpolate!(x, xh, Val{true})

    for i in 1:smooth2
        apply!(x, bc)
        gauss_seidel!(x, b, 1.0, conn)
    end

    x
end


function fmg!(conf::MGConfig, x, b, ws)
    @assert size(x) == size(b)
    @unpack bc, conn, levels, tolerance, s = conf

    apply!(x, bc)

    # Note that s appears only here; in the rest of the code we assume s=1.
    residual!(ws.res[1], x, b, s, conn)
    eps = norm(ws.res[1]) / sqrt(prod(innersize(x)))

    #@show eps
    if (s * tolerance > eps)
        return false        
    end

    
    for k in 1:levels
        restrict!(ws.res[k + 1], ws.res[k])
    end

    # Todo: change this to a proper solver
    z = ws.sol[levels + 1]
    bt = ws.res[levels + 1]

    ws.btop .= bt[inranges(bt)...][:]
    
    z[inranges(z)...][:] .= ws.mat \ ws.btop
    
    
    # for i in 1:10
    #     apply!(z, bc)
    #     gauss_seidel!(z, ws.res[levels + 1], 1.0, conn)
    # end        


    for k in 1:levels
        z1 = ws.sol[levels - k + 1]
        z1 .= 0
        
        apply!(z, bc)
        interpolate!(z1, z)

        mgv!(conf, z1, ws.res[levels - k + 1], levels - k, ws)

        z = z1
    end

    x .+= z
    apply!(x, bc)

    return true
end


"""
   Solve the Poisson equation Lx + s b = 0 where L is the discrete Laplace
   operator with grid size h=1.  You can change the grid size by 
   using s=h^2.  If you want to compute the electrostatic potential 
   solving ∇²ϕ = -q/ϵ0, use s = h^2 / ϵ0. 
"""
function solve(conf::MGConfig, x, b, ws)
    cont = true
    while cont
        cont = fmg!(conf, x, b, ws)
    end

    x
end


function checkerboard(a, parity)
    redblack(a, parity) do ind
        @inbounds a[ind] = oneunit(eltype(a))
    end
end

end # module
