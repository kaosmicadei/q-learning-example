using DelimitedFiles

include("../src/Warehouse.jl")
using .Warehouse

# Initialise the environment and agent
agent, env = readdlm("rewards.csv", ',', Int) |> initialize
train_agent!(agent, env)

# Tests
starting_states = [State(4,10), State(6,1), State(10,6), State(6,3)]

foreach(starting_states) do state
    display_path(agent, env, state)
    println('\n')
end
