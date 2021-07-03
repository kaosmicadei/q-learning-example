using DelimitedFiles
using Match

include("qlearning.jl")
include("policy.jl")

# Initialise the environment and agent
agent, env = readdlm(ARGS[1], ',') |> initialize

# Training
ϵ = 0.7
episodes = 5_000
for _ in 1:episodes
    state = get_starting_state(env)
    while !isterminal(state, env)
        action = get_next_action(agent, env, state, ϵ)
        next_state = get_next_state(env, state, action)
        update_qvalues!(agent, env, state, next_state, action)
        state = next_state
    end
end

# Tests
starting_states = [State(4,10), State(6,1), State(10,6), State(6,3)]

foreach(starting_states) do state
    path = get_shortest_path(agent, env, state)
    grid = zeros(Int8, size(env.rewards))
    grid[path] .= 99
    display(grid)
    println('\n')
end