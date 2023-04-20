using SDDP

Q = 5 # initial amount of money invested 
C_max = 5 # capacity of the storage
r = 0.05
l = 0.05
T = 10

function subproblem_builder(subproblem::Model, node::Int)
    # State variables
    @variable(subproblem, Z, SDDP.State, initial_value = -Q)
    @variable(subproblem, 0 <= C <= C_max, SDDP.State, initial_value = 0)
    # Control variables
    @variable(subproblem, -C_max <= U <= C_max) # units of electricity brought from the electricity market 
    # Random variables
    # the prices are low every other day. 
    @variable(subproblem, total_price)

    @constraint(subproblem,price,Z.out == (1+r)*Z.in - 1*U)

    Ω1 = [1.0, 1.5, 2.0]
    Ω2 = [2.5, 3.0, 3.5]
    P = [1 / 3, 1 / 3, 1 / 3]
    if node%2 == 0
        SDDP.parameterize(subproblem, Ω1, P) do ω
            return JuMP.set_normalized_coefficient(price,U, ω)
        end
    else
        SDDP.parameterize(subproblem, Ω2, P) do ω
            return JuMP.set_normalized_coefficient(price,U, ω)
        end
    end

    # Transition function and constraints
    @constraints(
        subproblem,
        begin
            C.out == (1 - l)*C.in + U # balance equation for the storage
        end
    )
    # Stage-objective
    if node == T
        @stageobjective(subproblem, Z.out) # maximize the final amount of money of the investor 
    else
        @stageobjective(subproblem,0.0)
    end
    return subproblem
end

using Gurobi

model = SDDP.LinearPolicyGraph(
    subproblem_builder;
    stages = T,
    sense = :Max,
    upper_bound = 1000.0,
    optimizer = Gurobi.Optimizer,
)

SDDP.train(model; iteration_limit = 10)
