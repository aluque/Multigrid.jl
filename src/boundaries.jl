# Implementations of boundary conditions

export LeftBnd, RightBnd, BottomBnd, TopBnd, Neumann, Dirichlet

abstract type AbstractBoundary end

struct LeftBnd <: AbstractBoundary end
struct RightBnd <: AbstractBoundary end
struct TopBnd <: AbstractBoundary end
struct BottomBnd <: AbstractBoundary end
struct FrontBnd <: AbstractBoundary end
struct BackBnd <: AbstractBoundary end

ghost2(a, ::BottomBnd)  = @view a[0, :]
ghost2(a, ::TopBnd)     = @view a[end, :]
ghost2(a, ::LeftBnd)    = @view a[:, 0]
ghost2(a, ::RightBnd)   = @view a[:, end]

ghost3(a, ::BottomBnd)  = @view a[0, :, :]
ghost3(a, ::TopBnd)     = @view a[end, :, :]
ghost3(a, ::LeftBnd)    = @view a[:, 0, :]
ghost3(a, ::RightBnd)   = @view a[:, end, :]
ghost3(a, ::FrontBnd)   = @view a[:, :, 0]
ghost3(a, ::BackBnd)    = @view a[:, :, end]

valid2(a, ::BottomBnd)  = @view a[1, :]
valid2(a, ::TopBnd)     = @view a[end - 1, :]
valid2(a, ::LeftBnd)    = @view a[:, 1]
valid2(a, ::RightBnd)   = @view a[:, end - 1]

valid3(a, ::BottomBnd)  = @view a[1, :, :]
valid3(a, ::TopBnd)     = @view a[end - 1, :, :]
valid3(a, ::LeftBnd)    = @view a[:, 1, :]
valid3(a, ::RightBnd)   = @view a[:, end - 1, :]
valid3(a, ::FrontBnd)   = @view a[:, :, 1]
valid3(a, ::BackBnd)    = @view a[:, :, end - 1]


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

@inline function setbc!(a, b::AbstractBoundary, c::AbstractLinearCondition)
    ghost(a, b) .= coef(c) .* valid(a, b) .+ cons(c)
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

