"""
$(TYPEDEF)

Type for dispatching Hermite polynomials, which are orthogonal with respect to the standard normal distribution.
"""
abstract type HermitePolynomials <: OrthogonalPolynomialType end

"""
$(TYPEDSIGNATURES)

Returns the recurrence coefficients `(a, b, c)` for the k-th Hermite polynomial, corresponding to the three-term recurrence relation:

    H_{k}(x) = (a_k x - b_k) H_{k-1}(x) - c_k H_{k-2}(x)

For Hermite polynomials:
- `a = 0`
- `b = 1`
- `c = k`
- `H_{-1} = 0`
- `H_0 = 1`
"""
recurrence_coefficients(::Type{HermitePolynomials}, k::Integer) =
    0, 1, k

issymmetric(::Type{HermitePolynomials}) = true

"""
$(TYPEDSIGNATURES)

Returns the norm of the k-th Hermite polynomial, i.e.,

    ||H_k|| = sqrt(k!)
"""
norms(::Type{HermitePolynomials}, k) = sqrt.(factorial.(big(k)))

"""
$(TYPEDSIGNATURES)

Returns the probability distribution associated with the Hermite polynomials, which is the standard normal distribution `Normal(0, 1)`.
"""
distribution(::Type{HermitePolynomials}) = Normal(0, 1)
