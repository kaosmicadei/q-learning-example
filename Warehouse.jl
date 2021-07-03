using DelimitedFiles
using Match

# Generic abstraction of a state
State = CartesianIndex

# Actions are encoded as index
Action = Int

# Define an environment
struct Environment
    rewards::Array{Int, N} where N
    actions::Int
end

struct Agent
    qvalues::Array{Float32, N} where N
end
Agent(e::Environment) = let dims = (size(e.rewards)..., e.actions)
    zeros(Float32, dims) |> Agent
end

function update_qvalues!(a::Agent, e::Environment, state::State, next_state::State, action::Action; α=0.9, γ=0.9)
    # α : learning rate
    # γ : discount factor
    reward = e.rewards[next_state]
    oldvalue = a.qvalues[state, action]
    optimal_action = maximum(a.qvalues[next_state,:])
    newvalue = reward + γ * optimal_action
    a.qvalues[state, action] = (1-α) * oldvalue + α * newvalue
end

# # Determine if the current state is terminal or if the agent can keep moving
isterminal(state::State, e::Environment) = e.rewards[state] != -1.0

# # Starting from a random state
get_starting_state(e::Environment) = findall(s -> s == -1, e.rewards) |> rand

get_next_action(a::Agent, e::Environment, state::State, ϵ) = rand() < ϵ ? argmax(a.qvalues[state,:]) : rand(1:e.actions)

function get_next_state(e::Environment, state::State, action::Action)
    move = @match action begin
        1 => State(-1,0)    # up
        2 => State(0,1)     # right
        3 => State(1,0)     # down
        4 => State(0,-1)    # left
    end
    try
        next = state + move
        checkbounds(e.rewards, next)
        next
    catch
        state
    end
end

function get_shortest_path(a::Agent, e::Environment, state::State)
  path = [state]
  while !isterminal(state, e)
    action = get_next_action(a, e, state, 1.0)
    state = get_next_state(e, state, action)
    push!(path, state)
  end
  path
end

# Initialise the environment and agent
env = Environment(readdlm("rewards.csv", ','), 4)
agent = Agent(env)

# Training part
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