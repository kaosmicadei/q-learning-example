# Determine if the current state is terminal or if the agent can keep moving
isterminal(state::State, e::Environment) = e.rewards[state] != -1.0

# Starting from a random state
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
