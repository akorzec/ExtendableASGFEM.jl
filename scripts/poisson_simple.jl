#=
Minimal example for a stochastic Poisson problem solved with stochastic Galerkin FEM.

This script follows the standard workflow of the package:
1. choose the stochastic coefficient,
2. build the spatial grid and stochastic basis,
3. solve the SGFEM problem,
4. estimate the error by Monte Carlo sampling,
5. plot the stochastic modes.

Possible problem types:
- `PoissonProblemPrimal`: standard Poisson problem with linear diffusion coefficient `a`
- `LogTransformedPoissonProblemPrimal`: log-transformed Poisson problem with coefficient `exp(a)`
- `LogTransformedPoissonProblemDual`: dual formulation of the log-transformed problem

Typical usage:
    include("scripts/poisson_simple.jl")
    sol = PoissonSimple.main(
        problem = LogTransformedPoissonProblemPrimal,
        domain = "square",
        nrefs = 3,
        order = 2,
        decay = 2.0,
        mean = 0.0,
    )
=#

module PoissonSimple

using ExtendableASGFEM
using ExtendableFEM
using ExtendableFEMBase
using ExtendableGrids
using GridVisualize
using UnicodePlots
using Term

function coefficient_for_problem(problem; decay, mean)
    τ = (problem <: PoissonProblemPrimal) ? 0.9 : 1.0
    if problem <: PoissonProblemPrimal
        @assert mean >= 1 "coefficient mean value needs to be at least 1 to ensure ellipticity"
    end
    return StochasticCoefficientCosinus(; τ = τ, decay = decay, mean = mean)
end

function make_grid(domain::String, nrefs)
    if domain == "square"
        return uniform_refine(grid_unitsquare(Triangle2D), nrefs)
    elseif domain == "lshape"
        return uniform_refine(grid_lshape(Triangle2D), nrefs)
    else
        error("unknown domain: $domain")
    end
end

function make_stochastic_basis(problem, initial_modes)
    multi_indices = Array{Array{Int, 1}, 1}(initial_modes)
    prepare_multi_indices!(multi_indices)
    M = maximum(length.(multi_indices))
    polynomial_family = problem <: PoissonProblemPrimal ? LegendrePolynomials : HermitePolynomials
    ansatz_degree = maximum([maximum(multi_indices[k]) for k in 1:length(multi_indices)]) + 4
    stochastic_basis = TensorizedBasis(
        polynomial_family,
        M,
        ansatz_degree,
        2 * ansatz_degree,
        2 * ansatz_degree;
        multi_indices = multi_indices,
    )
    return stochastic_basis, multi_indices
end

function make_fem_spaces(problem, xgrid, order)
    if problem <: LogTransformedPoissonProblemDual
        FEType = [HDIVRTk{2, order}, order == 0 ? L2P0{1} : H1Pk{1, 2, order}]
        FES = [FESpace{FEType[1]}(xgrid), FESpace{FEType[2]}(xgrid; broken = true)]
        unames = ["p", "u"]
    else
        FEType = H1Pk{1, 2, order}
        FES = FESpace{FEType}(xgrid)
        unames = ["u"]
    end
    return FES, unames
end

function main(;
        problem = PoissonProblemPrimal, # one of: PoissonProblemPrimal, LogTransformedPoissonProblemPrimal, LogTransformedPoissonProblemDual
        nrefs = 3,      # number of uniform refinements of the initial grid
        order = 2,      # polynomial order of the FE spaces
        decay = 2.0,    # decay factor for the random coefficient
        mean = problem == PoissonProblemPrimal ? 1.0 : 0.0, # mean value of the coefficient
        domain = "square",  # domain, e.g., "square" or "lshape"
        initial_modes = [[0], [1, 0], [0, 1], [2, 0], [0, 0, 1]], # initial multi-indices for the stochastic basis
        f! = (result, qpinfo) -> (result[1] = 1), # right-hand side
        use_iterative_solver = true,
        calculate_error = true, # compute Monte Carlo error estimates (set false for quick solver-only runs)
        Plotter = UnicodePlots,
    )

    ## build the stochastic coefficient
    C = coefficient_for_problem(problem; decay = decay, mean = mean)

    ## build the spatial mesh
    xgrid = make_grid(domain, nrefs)

    ## build the stochastic basis
    tensor_basis, multi_indices = make_stochastic_basis(problem, initial_modes)

    ## build the FE spaces
    FES, unames = make_fem_spaces(problem, xgrid, order)

    ## create the solution vector
    sol = SGFEVector(FES, tensor_basis; active_modes = 1:length(multi_indices), unames = unames)

    ## solve the stochastic Galerkin problem
    @info "Solving..."
    solve!(problem, sol, C; rhs = f!, use_iterative_solver = use_iterative_solver)

    ## compute a Monte Carlo reference error estimate
    if calculate_error
        weightederrorH1, weightederrorL2, uniformerrorH1, uniformerrorL2 = calculate_sampling_error(
            sol,
            C;
            problem = problem,
            rhs = f!,
            order = order + 1,
            nsamples = 50,
        )

        @info "RESULTS
        || ∇(u-u_h) || (w,u) = $(sqrt(weightederrorH1[end])), $(sqrt(uniformerrorH1[end]))
        || u - u_h || (w,u) = $(sqrt(weightederrorL2[end])), $(sqrt(uniformerrorL2[end]))"
    end

    ## plot the stochastic modes if a plotting backend was requested
    if !isnothing(Plotter)
        p = plot_modes(sol; Plotter = Plotter, ncols = 4)
        display(p)
    end

    return sol
end

end # module
