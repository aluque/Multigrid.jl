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

    
@inline ghost2(a, ::BottomBnd)  = @view a[gbegin(a, 1), :]
@inline ghost2(a, ::TopBnd)     = @view a[gend(a, 1), :]
@inline ghost2(a, ::LeftBnd)    = @view a[:, gbegin(a, 2)]
@inline ghost2(a, ::RightBnd)   = @view a[:, gend(a, 2)]

@inline ghost3(a, ::BottomBnd)  = @view a[gbegin(a, 1), :, :]
@inline ghost3(a, ::TopBnd)     = @view a[gend(a, 1), :, :]
@inline ghost3(a, ::LeftBnd)    = @view a[:, gbegin(a, 2), :]
@inline ghost3(a, ::RightBnd)   = @view a[:, gend(a, 2), :]
@inline ghost3(a, ::FrontBnd)   = @view a[:, :, gbegin(a, 3)]
@inline ghost3(a, ::BackBnd)    = @view a[:, :, gend(a, 3)]

@inline valid2(a, ::BottomBnd)  = @view a[1, :]
@inline valid2(a, ::TopBnd)     = @view a[end - ng(a, 1), :]
@inline valid2(a, ::LeftBnd)    = @view a[:, 1]
@inline valid2(a, ::RightBnd)   = @view a[:, end - ng(a, 2)]

@inline valid3(a, ::BottomBnd)  = @view a[1, :, :]
@inline valid3(a, ::TopBnd)     = @view a[end - ng(a, 1), :, :]
@inline valid3(a, ::LeftBnd)    = @view a[:, 1, :]
@inline valid3(a, ::RightBnd)   = @view a[:, end - ng(a, 2), :]
@inline valid3(a, ::FrontBnd)   = @view a[:, :, 1]
@inline valid3(a, ::BackBnd)    = @view a[:, :, end - ng(a, 3)]


dim(::BottomBnd) = 1
dim(::TopBnd) = 1
dim(::LeftBnd) = 2
dim(::RightBnd) = 2
dim(::FrontBnd) = 3
dim(::BackBnd) = 3

targetind(rng, ::BottomBnd) = (first(rng) - 1, first(rng))
targetind(rng, ::TopBnd) = (last(rng) + 1, last(rng))
targetind(rng, ::LeftBnd) = (first(rng) - 1, first(rng))
targetind(rng, ::RightBnd) = (last(rng) + 1, last(rng))
targetind(rng, ::FrontBnd) = (first(rng) - 1, first(rng))
targetind(rng, ::BackBnd) = (last(rng) + 1, last(rng))
                              
@generated function ghost(a::AbstractArray{T, N}, b::AbstractBoundary) where {T, N}
    if N == 2
        expr = quote
            ghost2(a, b)
        end
    elseif N == 3
        expr = quote
            ghost3(a, b)
        end
    else
        throw(ArgumentError("Only arrays with 2, 3 dimensions allowed"))
    end
end


@generated function valid(a::AbstractArray{T, N}, b::AbstractBoundary) where {T, N}
    if N == 2
        expr = quote
            valid2(a, b)
        end
    elseif N == 3
        expr = quote
            valid3(a, b)
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

@inline function setbc!(a::AbstractArray{T, N}, b::AbstractBoundary,
                        c::AbstractLinearCondition) where {T, N}
    # All these contortions are required for CUDA compatibility.
    # CUDA.jl uses scalar indexes when we operate with views into OffsetArrays.
    # This is avoided by transforming into a view of the underlying CuArray
    l = ghost(a, b)
    indl = ntuple(i->l.indices[i] .- a.offsets[i], Val(N))

    v = valid(a, b)
    indv = ntuple(i->v.indices[i] .- a.offsets[i], Val(N))
    p = parent(a)
    @views p[indl...] .= coef(c) .* p[indv...] .+ cons(c)
end

@generated function apply!(a, bc::T) where {T}
    L = fieldcount(T)
    out = quote end 

    for i in 1:L
        push!(out.args,
              quote
              setbc!(a, bc[$i][1], bc[$i][2])
              end
              )
    end
    push!(out.args, :(return nothing))

    out
end


@inline function matcoef(rngs, ind, b::AbstractBoundary,
                         c::AbstractLinearCondition)
    i1, i2 = targetind(rngs[dim(b)], b)
    if ind[dim(b)] == i1
        return true, Base.setindex(ind, i2, dim(b)), coef(c)
    end
    return false, ind, 1
end

