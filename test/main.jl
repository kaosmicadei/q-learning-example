using DelimitedFiles
using Base.Threads

include("../src/Warehouse.jl")
using .Warehouse

# Initialise the environment and agent
ae1 = readdlm("rewards.csv", ',', Int) |> initialize
ae2 = readdlm("rewards.csv", ',', Int) |> initialize
ae3 = readdlm("rewards.csv", ',', Int) |> initialize
ae4 = readdlm("rewards.csv", ',', Int) |> initialize

@threads for (a,e) in [ae1, ae2, ae3, ae4]
    train_agent!(a, e; ϵ=0.6)
end

# Tests
# starting_states = [State(4,10), State(6,1), State(10,6), State(6,3)]
starting_states = [State(10, 6)]

println("Agent: 1")
foreach(starting_states) do state
    display_path(ae1..., state)
    println('\n')
end

println("Agent: 2")
foreach(starting_states) do state
    display_path(ae2..., state)
    println('\n')
end

println("Agent: 3")
foreach(starting_states) do state
    display_path(ae3..., state)
    println('\n')
end

println("Agent: 4")
foreach(starting_states) do state
    display_path(ae4..., state)
    println('\n')
end