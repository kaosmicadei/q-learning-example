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
    @inbounds a.qvalues[state, action] = (1-α) * oldvalue + α * newvalue
end
