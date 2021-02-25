# Implementations of boundary conditions

export LeftBnd, RightBnd, BottomBnd, TopBnd, Neumann, Dirichlet

abstract type AbstractBoundary end

struct LeftBnd <: AbstractBoundary end
struct RightBnd <: AbstractBoundary end
struct TopBnd <: AbstractBoundary end
struct BottomBnd <: AbstractBoundary end
struct FrontBnd <: AbstractBoundary end
struct BackBnd <: AbstractBoundary end

ng(a, d) = 1 - first(axes(a)[d])
gbegin(a, d) = 0
gend(a, d) = last(axes(a)[d]) - ng(a, d) + 1

@inline setbc2!(a, ::BottomBnd, c) = @views a[1, :]   .= coef(c) .* a[2, :]       .+ cons(c)
@inline setbc2!(a, ::TopBnd, c)    = @views a[end, :] .= coef(c) .* a[end - 1, :] .+ cons(c)
@inline setbc2!(a, ::LeftBnd, c)   = @views a[:, 1]   .= coef(c) .* a[:, 2]       .+ cons(c)
@inline setbc2!(a, ::RightBnd, c)  = @views a[:, end] .= coef(c) .* a[:, end - 1] .+ cons(c)

    
@inline ghost2(g, a, ::BottomBnd)  = @view a[begin + g - 1, :]
@inline ghost2(g, a, ::TopBnd)     = @view a[end - g + 1, :]
@inline ghost2(g, a, ::LeftBnd)    = @view a[:, begin + g - 1]
@inline ghost2(g, a, ::RightBnd)   = @view a[:, end - g + 1]

@inline ghost3(g, a, ::BottomBnd)  = @view a[begin + g - 1, :, :]
@inline ghost3(g, a, ::TopBnd)     = @view a[end - g + 1, :, :]
@inline ghost3(g, a, ::LeftBnd)    = @view a[:, begin + g - 1, :]
@inline ghost3(g, a, ::RightBnd)   = @view a[:, end - g + 1, :]
@inline ghost3(g, a, ::FrontBnd)   = @view a[:, :, begin + g - 1]
@inline ghost3(g, a, ::BackBnd)    = @view a[:, :, end - g + 1]

@inline valid2(g, a, ::BottomBnd)  = @view a[begin + g, :]
@inline valid2(g, a, ::TopBnd)     = @view a[end - g, :]
@inline valid2(g, a, ::LeftBnd)    = @view a[:, begin + g]
@inline valid2(g, a, ::RightBnd)   = @view a[:, end - g]

@inline valid3(g, a, ::BottomBnd)  = @view a[begin + g, :, :]
@inline valid3(g, a, ::TopBnd)     = @view a[end - g, :, :]
@inline valid3(g, a, ::LeftBnd)    = @view a[:, begin + g, :]
@inline valid3(g, a, ::RightBnd)   = @view a[:, end - g, :]
@inline valid3(g, a, ::FrontBnd)   = @view a[:, :, begin + g]
@inline valid3(g, a, ::BackBnd)    = @view a[:, :, end - g]


dim(::BottomBnd) = 1
dim(::TopBnd) = 1
dim(::LeftBnd) = 2
dim(::RightBnd) = 2
dim(::FrontBnd) = 3
dim(::BackBnd) = 3

dirind(::BottomBnd) = -1
dirind(::TopBnd) = 1
dirind(::LeftBnd) = -1
dirind(::RightBnd) = 1
dirind(::FrontBnd) = -1
dirind(::BackBnd) = 1

targetind(rng, ::BottomBnd) = (first(rng) - 1, first(rng))
targetind(rng, ::TopBnd)    = (last(rng)  + 1, last(rng))
targetind(rng, ::LeftBnd)   = (first(rng) - 1, first(rng))
targetind(rng, ::RightBnd)  = (last(rng)  + 1, last(rng))
targetind(rng, ::FrontBnd)  = (first(rng) - 1, first(rng))
targetind(rng, ::BackBnd)   = (last(rng)  + 1, last(rng))
                              
@generated function ghost(g, a::AbstractArray{T, N}, b::AbstractBoundary) where {T, N}
    if N == 2
        expr = quote
            ghost2(g, a, b)
        end
    elseif N == 3
        expr = quote
            ghost3(g, a, b)
        end
    else
        throw(ArgumentError("Only arrays with 2, 3 dimensions allowed"))
    end
end


@generated function valid(g, a::AbstractArray{T, N}, b::AbstractBoundary) where {T, N}
    if N == 2
        expr = quote
            valid2(g, a, b)
        end
    elseif N == 3
        expr = quote
            valid3(g, a, b)
        end
    else
        throw(ArgumentError("Only arrays with 2, 3 dimensions allowed"))
    end
end


abstract type AbstractLinearCondition end

struct Dirichlet <: AbstractLinearCondition end
struct Neumann <: AbstractLinearCondition end

@inline coef(c::Dirichlet) = -1
@inline coef(c::Neumann) = 1
@inline cons(c::Dirichlet) = 0
@inline cons(c::Neumann) = 0

@inline function setbc!(g, a::AbstractArray{T, N},
                        b::AbstractBoundary,
                        c::AbstractLinearCondition) where {T, N}
    l = ghost(g, a, b)
    v = valid(g, a, b)

    l .= coef(c) .* v .+ cons(c)
end

@generated function apply!(g, a, bc::T) where {T}
    L = fieldcount(T)
    out = quote end 

    for i in 1:L
        push!(out.args,
              quote
              setbc!(g, a, bc[$i][1], bc[$i][2])
              end
              )
    end
    push!(out.args, :(return nothing))

    out
end

"""
Modifies the rhs of the Poisson equation to satisfy an inhomogeneous boundary 
condition with independent term in array `val`.  Note that if you want
Neumann inhomogeneous condition to the derivative of ϕ you should include
a factor h in `val`.
"""
function setinhom!(conf::MGConfig,
                   b::AbstractArray{T, N},
                   bnd::AbstractBoundary,
                   val::AbstractArray) where {T, N}
    @unpack s, g, conn = conf
    v = valid(g, b, bnd)
    
    z = zero(CartesianIndex{N})
    I = Base.setindex(z, v.indices[dim(bnd)], dim(bnd))
    D = Base.setindex(z, dirind(bnd), dim(bnd))
    @show conn(g, I, D, 2 / s)
    
    v .+= conn(g, I, D, 2 / s) .* val
end
    
@inline function matcoef(rngs, ind, b::AbstractBoundary,
                         c::AbstractLinearCondition)
    i1, i2 = targetind(rngs[dim(b)], b)
    if ind[dim(b)] == i1
        return true, Base.setindex(ind, i2, dim(b)), coef(c)
    end
    return false, ind, 1
end

