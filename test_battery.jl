using SDDP
using Ipopt
using Distributed 

function main_SDDP(file_name = "data.txt", result_file = "test_default.txt", iteration_limit = 30, parallel = false, nb_processors = 5, mu = 1)

    #load data

    include(file_name)


    # create the Markovian graph

    global graph = SDDP.Graph((0,exp(moy_ln[1])))

    for t in 1:T
        for x_val in x
            SDDP.add_node(graph,(t,exp(moy_ln[t+1]+mu*x_val)))
        end
    end

    for i in 1:10
        SDDP.add_edge(graph, (0,exp(moy_ln[1])) => (1,exp(moy_ln[2] + mu*x[i])), P0[i])
    end

    for i in 1:10
        for j in 1:10
            for t in 2:T
                SDDP.add_edge(graph, (t-1,exp(moy_ln[t] + mu*x[i])) => (t,exp(moy_ln[t+1] + mu*x[j])), P[i][j])
            end
        end
    end 


    # create the model

    model = SDDP.PolicyGraph(
        graph,  # <--- New stuff
        sense = :Min,
        lower_bound = 0.0,
        optimizer = Ipopt.Optimizer,
    ) do sp, node
        t, price = node 
        @variable(sp, Z, SDDP.State, initial_value = -Q/sc)
        @variable(sp, 0 <= C <= C_max, SDDP.State, initial_value = 0)
        @variable(sp, -C_max <= U <= C_max)
        @variable(sp, 0<= utility, SDDP.State, initial_value = exp(rho*Q/sc)/rho)
        @constraint(sp,Z.out == (1+r)*Z.in - price*U/sc)
        @constraint(sp,C.out == (1 - l)*C.in + U)

        # NL constraints 

        @NLconstraint(sp, utility.out == exp(-rho*Z.out)/rho)
        

        if t == T
            @stageobjective(sp, utility.out) # maximize the final amount of money of the investor 
            #@stageobjective(sp, -Z.out)
        else
            @stageobjective(sp,0.0)
        end
    end


    # train the model 

    if parallel
        Distributed.addprocs(nb_processors)

        SDDP.train(model; iteration_limit, parallel_scheme = SDDP.Asynchronous(), log_file = result_file)
    else
        SDDP.train(model; iteration_limit, log_file = result_file)
    end


    # compute the price of the battery

    utility = SDDP.calculate_bound(model)
    beta = (sc/rho)*log(1/(rho*utility))
    alpha = (1+r)^T
    price = beta/alpha 
    
    return model, price 

end


function simulations(model, nb_simulations = 5, parallel = false)

    if parallel 
        simulations = SDDP.simulate(model,nb_simulations,[:Z, :U], parallel_scheme = SDDP.Asynchronous())
    else
        simulations = SDDP.simulate(model,nb_simulations,[:Z, :U])
    end

    plt = SDDP.SpaghettiPlot(simulations)

    SDDP.add_spaghetti(plt; title = "Z") do data
        return data[:Z].out
    end


    SDDP.add_spaghetti(plt; title = "U") do data
        return data[:U]
    end


    SDDP.add_spaghetti(plt; title = "price") do data
        return (sc/data[:U])*((1+r)*data[:Z].in - data[:Z].out)
    end

    SDDP.plot(plt, "spaghetti_plot.html")

end 

