module Warehouse

export State, initialize, train_agent!, display_path

using Match

include("qlearning.jl")

# Determine if the current state is terminal or if the agent can keep moving.
isterminal(state::State, e::Environment) = e.rewards[state] != -1.0

# Return a valid initial state.
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

function initialize(rewards)
    e = Environment(rewards, 4)
    a = Agent(e)
    a, e
end

function train_agent!(a::Agent, e::Environment; ϵ=0.7, episodes=1_000)
    for _ in 1:episodes
        state = get_starting_state(e)
        while !isterminal(state, e)
            action = get_next_action(a, e, state, ϵ)
            next_state = get_next_state(e, state, action)
            update_qvalues!(a, e, state, next_state, action)
            state = next_state
        end
    end
end

function display_path(a::Agent, e::Environment, s::State)
    path = get_shortest_path(a, e, s)
    grid = zeros(Int8, size(e.rewards))
    grid[path] .= 99
    display(grid)
end

end # module