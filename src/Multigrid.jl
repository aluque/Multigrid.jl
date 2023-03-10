"""
Multigrid code for the solution of Poisson's equation.

This code is valid both for 2d and 3d and for (multi-threaded) CPU or GPU
execution.

(c) Alejandro Luque, IAA-CSIC (2021)
"""

module Multigrid
using Polyester
using OffsetArrays
using LinearAlgebra
using SparseArrays
using Parameters

export MGConfig
export LeftBnd, RightBnd, TopBnd, BottomBnd, FrontBnd, BackBnd
export Dirichlet, Neumann
export CartesianConnector, CylindricalConnector

""" A connector is a type that allows us for example to implement
cylindrical symmetry in a generic code without a performance overhead.

The connector must be able to compute from a given grid coordinate and
a stencil shift a transformation of the laplacian term (generally a 
multiplication).
"""
abstract type AbstractConnector end

struct CartesianConnector <: AbstractConnector end
(::CartesianConnector)(g, ::CartesianIndex, ::CartesianIndex, x) = x

# For performance we store the cylindrical dimension in a type parameters
struct CylindricalConnector{D} <: AbstractConnector end

function (::CylindricalConnector{D})(g, c::CartesianIndex, d::CartesianIndex, x) where D
    x * (1.0 + d[D] / (2 * (c[D] - g) - 1))
end

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

    " Number of ghost cells in each boundary (must be at least 1)."
    g::Int = 1

    " Maximum number of iterations."
    maxiter::Int = 50
    
    " Verbosity level"
    verbosity::Int = 0

    " Error if not converging. "
    convergenceerror::Bool = true
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


include("redblack.jl")
include("boundaries.jl")
include("cuda.jl")

function inranges(g, a::AbstractArray{T, N}) where {T, N}
    ntuple(n -> (firstindex(a, n) + g):(lastindex(a, n) - g), Val(N))
end

function inends(g, a::AbstractArray{T, N}) where {T, N}
    ntuple(i -> lastindex(a, i) - firstindex(a, i) + 1 - 2g, Val(N))
end

innerindices(g, a) = CartesianIndices(inranges(g, a))
innersize(g, a) = length.(inranges(g, a))

# Wrapper around the zeros function that works also for CUDA arrays
function simzeros(g, a)
    s = similar(a)
    fill!(s, 0)
    s
end

function simcoarser(g, a::AbstractArray{T, N}) where {T, N}
    zeros(eltype(a), ntuple(i->(size(a, i) - 2g) ÷ 2 + 2g, Val(N)))
end


function simfiner(g, a::AbstractArray{T, N}) where {T, N}
    zeros(eltype(a), ntuple(i->(size(a, i) - 2g) * 2 + 2g, Val(N)))
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


@inline function laplacian(g, u, ind, st, c::AbstractConnector)
    @inbounds s = -length(st) * u[ind]
    for j in st
        @inbounds s += c(g, ind, j, u[ind + j])
    end
    s
end


# @inline laplacian(u, ind, st) = laplacian(u, ind, st, CartesianConnector())

    
"""
   Update the potential `u` with Gauss-Seidel using the source `b`.
   `ω` is an over-relaxation parameter.
"""
function gauss_seidel!(g, u, b, ω, c::AbstractConnector)
    @assert size(u) == size(b)
    st = lplstencil(u)

    for parity in (false, true)
        redblack(g, u, parity) do ind
            l = laplacian(g, u, ind, st, c)
            @inbounds u[ind] += ω * (l + b[ind]) / length(st)
        end
    end
end
gauss_seidel!(g, u, b, ω) = gauss_seidel!(g, u, b, ω, CartesianConnector())


"""
   Computes the residual of Laplace operator acting on u with rhs -b.
   The function computes Lu + s b where L is the discrete laplace operator.
"""
function residual!(g, r, u, b, s, c::AbstractConnector)
    st = lplstencil(u)
    
    @batch for ind in innerindices(g, u)
        l = laplacian(g, u, ind, st, c)
        @inbounds r[ind] = s * b[ind] + l
    end
end

residual!(g, r, u, b, c::AbstractConnector) = residual!(g, r, u, b, 1.0, c)
residual!(g, r, u, b, s::Real) = residual!(g, r, u, b, s, CartesianConnector())


function residualnorm(g, u, b, c::AbstractConnector)
    r = simzeros(g, u)
    residual!(g, r, u, b, c)

    norm(r)
end

residualnorm(g, u, b) = residualnorm(g, u, b, CartesianConnector())



"""
    Restrict coarse grid `rh` into `r`.
"""
function restrict!(g, rh, r)
    st = cubestencil(r)
    
    @batch for irh in innerindices(g, rh)
        ir = 2 * (irh - (g + 1) * oneunit(irh)) + (g + 1) * oneunit(irh)
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
function interpolate!(g, r, rh, update::Type{Val{V}}=Val{false}) where {V}
    st = cubestencil(r)
    weights = binterpweights(st)
    
    @batch for irh in innerindices(g, rh)
        ir = 2 * (irh - (g + 1) * oneunit(irh)) + (g + 1) * oneunit(irh)
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


function buildmat(g, x, bc)
    rngs = map(r -> r .- g, inranges(g, x))
    G = CartesianIndex(ntuple(_->g, ndims(x)))
    
    cinds = innerindices(g, x) .- G
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

"""
Allocate space for the grid hierarchy needed to solve Poisson's equation in grid
`x` (but note that the contents of `x` do not need to be filled yet).
Return a Workspace with the neccesary allocations.
"""
function allocate(conf, x)
    @unpack levels, bc, g = conf
    
    res = typeof(x)[]
    sol = typeof(x)[]
    res1 = typeof(x)[]
    
    push!(res, simzeros(g, x))
    push!(sol, simzeros(g, x))
    push!(res1, simzeros(g, x))

    for i in 1:levels
        push!(res, simcoarser(g, last(res)))
        push!(res1, simcoarser(g, last(res1)))
        push!(sol, simcoarser(g, last(sol)))
    end

    btop = vec(zeros(innersize(g, sol[end])...))
    utop = vec(zeros(innersize(g, sol[end])...))
    
    mat = buildmat(g, sol[end], bc)
    Workspace(res, res1, sol, btop, utop, mat)
end



""" 

Multigrid V-cycle. Improves a guess for the discrete Poisson 
equation A.x == -b  where A is the discrete Laplacian operator with h=1.
    
"""
function mgv!(conf::MGConfig, x, b, level, ws)
    @assert size(x) == size(b)
    @unpack bc, conn, smooth1, smooth2, levels, g = conf
    
    st = lplstencil(x)
    
    if level == levels 
        for i in 1:50
            apply!(g, x, bc)
            gauss_seidel!(g, x, b, 1.0, conn)
        end
        return x
    end

    for i in 1:smooth1
        apply!(g, x, bc)
        gauss_seidel!(g, x, b, 1.0, conn)
    end

    # We need extra space here to avoid clashing with the residuals
    # computed in fmg!
    r = ws.res1[level + 1]
    
    apply!(g, x, bc)
    residual!(g, r, x, b, conn)
    
    rh = ws.res[level + 2]
    
    apply!(g, r, bc)
    restrict!(g, rh, r)

    xh = ws.sol[level + 2]
    xh .= 0
    
    mgv!(conf, xh, rh, level + 1, ws)
    interpolate!(g, x, xh, Val{true})

    for i in 1:smooth2
        apply!(g, x, bc)
        gauss_seidel!(g, x, b, 1.0, conn)
    end

    x
end


function fmg!(conf::MGConfig, x, b, ws)
    @assert size(x) == size(b)
    @unpack bc, conn, levels, tolerance, s, g, verbosity = conf

    apply!(g, x, bc)

    ws.res[1] .= 0
    
    # Note that s appears only here; in the rest of the code we assume s=1.
    residual!(g, ws.res[1], x, b, s, conn)

    eps = norm(ws.res[1]) / sqrt(prod(innersize(g, x)))

    verbosity < 2 || @info "Residual norm: $(eps)"
        
    if (s * tolerance > eps)
        verbosity < 2 || @info "Convergence achieved"
        return false        
    end
    
    for k in 1:levels
        restrict!(g, ws.res[k + 1], ws.res[k])
    end

    z = ws.sol[levels + 1]
    bt = ws.res[levels + 1]

    copyto!(ws.btop, vec(@view bt[inranges(g, bt)...]))

    inr = inranges(g, z)
    copyto!(@view(z[inr...]), reshape(ws.mat \ ws.btop, length.(inr)...))
    
    # for i in 1:10
    #     apply!(z, bc)
    #     gauss_seidel!(z, ws.res[levels + 1], 1.0, conn)
    # end        


    for k in 1:levels
        z1 = ws.sol[levels - k + 1]
        z1 .= 0
        
        apply!(g, z, bc)
        interpolate!(g, z1, z)

        mgv!(conf, z1, ws.res[levels - k + 1], levels - k, ws)

        z = z1
    end

    x .+= z
    apply!(g, x, bc)

    return true
end


"""
    Solve the Poisson equation Lx + s b = 0 where L is the discrete Laplace
    operator with grid size h=1.  You can change the grid size by 
    using s=h^2.  If you want to compute the electrostatic potential 
    solving ∇²ϕ = -q/ϵ0, use s = h^2 / ϵ0. 

    `x` and `b` are arrays that must contain enough space for `conf.g ≥ 1` ghost 
    cells.  For example to solve the Poisson equation in a MxN grid you can set
    `g = 1` and allocate arrays (M + 2) x (N + 2).  If you prefer t you can use
     OffsetArrays indexed as (0:M+1, 0:N+1) but then you should call
     `Multigrid.allocate` and `Multigrid.solve` with parent(x), parent(b).
"""
function solve(conf::MGConfig, x, b, ws)
    cont = true
    local iter = 0
    while cont && iter <= conf.maxiter
        cont = fmg!(conf, x, b, ws)
        iter += 1
    end
    conf.verbosity < 1 || @info "$iter iterations"

    if iter > conf.maxiter
        if conf.convergenceerror
            @error "Convergence failed with [maxiter=] $(conf.maxiter) iterations"
        else
            @warn "Convergence failed with [maxiter=] $(conf.maxiter) iterations"
        end
    end
    
    x
end


function checkerboard(a, parity)
    redblack(a, parity) do ind
        @inbounds a[ind] = oneunit(eltype(a))
    end
end

end # module
