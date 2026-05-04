module Example_Stokes

using ExtendableFEM
using ExtendableGrids
using GridVisualize
using CairoMakie

default_plotter!(CairoMakie)

nu = 10^(-1)

function f!(result, qpinfo)
    x = qpinfo.x[1]
    y = qpinfo.x[2]
    ddu1 = -4 * (2 * y - 1) * (3 * x^4 - 6 * x^3 + 3 * x^2 * (2 * y^2 - 2 * y + 1) + (1 - 6 * x) * (y - 1) * y)
    ddu2 = 4 * (2 * x - 1) * ((x - 1) * x * (6 * y^2 - 6 * y + 1) + 3 * (y - 1)^2 * y^2)
    dp1 = 5 * x^4
    dp2 = 5 * y^4
    result[1] = -nu * ddu1 + dp1
    result[2] = -nu * ddu2 + dp2
    return nothing
end

function g!(result, _)
    result[1] = 0
    return nothing
end

function main(; Plotter=default_plotter(), nrefs=6, kwargs...)
    # generate two problems definitions
    # one for velocity, one for pressure
    u = Unknown("u", name="velocity", symbol_ansatz="u", symbol_test="v", dim=2)
    p = Unknown("p", name="pressure", symbol_ansatz="p", symbol_test="q", dim=2)
    PDu = ProblemDescription("Stokes IPM - velocity update")
    assign_unknown!(PDu, u)
    assign_operator!(PDu, BilinearOperator([grad(u)]; factor=nu, store=true, kwargs...))
    assign_operator!(PDu, LinearOperator([div(u)], [id(p)]; factor=-1, store=true, kwargs...))
    assign_operator!(PDu, LinearOperator(f!, [id(u)]; kwargs...))
    assign_operator!(PDu, HomogeneousBoundaryData(u; regions=1:4))

    PDp = ProblemDescription("Stokes IPM - pressure update")
    assign_unknown!(PDp, p)
    assign_operator!(PDp, LinearOperator([id(p)], [div(u)]; store=true, kwargs...))
    assign_operator!(PDp, LinearOperator(g!, [id(p)]; kwargs...))

    # create grid
    xgrid = uniform_refine(grid_unitsquare(Triangle2D), nrefs)

    # create P2/P1 pair
    FETypes = (H1P2{2,2}, H1P1{1})

    # solve problem
    FES = [FESpace{FETypes[1]}(xgrid), FESpace{FETypes[2]}(xgrid)]
    sol = FEVector(FES, tags=[u, p])
    SC1 = SolverConfiguration(PDu; init=sol, maxiterations=1, target_residual=1.0e-8, constant_matrix=true, kwargs...)
    SC2 = SolverConfiguration(PDp; init=sol, maxiterations=1, target_residual=1.0e-1, constant_matrix=true, kwargs...)
    sol, nits = iterate_until_stationarity([SC1, SC2]; init=sol, kwargs...)
    @info "converged after $nits iterations"

    # plot
    plt = ExtendableFEM.plot([id(u), id(p)], sol; Plotter=Plotter, show=true)
    ExtendableFEM.save("test.png", plt)

    return sol, plt
end

end