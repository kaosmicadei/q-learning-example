using DelimitedFiles

# Generic abstraction of a state
State = CartesianIndex

# Possible actions the agent can execute.
@enum Action up=1 right down left

struct QLearning
  rewards::Matrix{Float64}
  values::Array{Float64,3}
end

QLearning(rewards) = let
  (rows, cols) = size(rewards)
  values = zeros(Float64, rows, cols, Action.size)
  QLearning(rewards, values)
end

function update_qvalues!(q::QLearning, state::State, nextState::State, action::Action; α=0.9, γ=0.9)
  # α : learning rate
  # γ : discount factor
  action_index = Int(action)
  reward = q.rewards[nextState]
  oldValue = q.values[state, action_index]
  optimalNextAction = maximum(q.values[nextState,:])
  newValue = reward + γ * optimalNextAction
  q.values[state, action_index] = (1-α) * oldValue + α * newValue
end

# Determine if the current state is terminal or if the agent can keep moving
isterminal(position::State, q::QLearning) = q.rewards[position] != -1.0

# Starting from a random location
getStartingLocation(q::QLearning) = let validPoints = findall(s -> s == -1, q.rewards)
  rand(validPoints)
end

getNextAction(q::QLearning, position::State, ϵ) = let
  index = rand() < ϵ ? argmax(q.values[position,:]) : rand(1:Action.size)
  Action(index)
end

function getNextLocation(q::QLearning, position::State, action::Action)
  move = if action == up;    State(-1,0)
     elseif action == right; State(0,1)
     elseif action == down;  State(1,0)
     elseif action == left;  State(0,-1)
     end
  next = position + move
  try
    checkbounds(q.values[:,:,1], next)
    next
  catch
    position
  end
end

function getShortestPath(q::QLearning, position::State)
  path = [position]
  while !isterminal(position, q)
    action = getNextAction(q, position, 1.0)
    position = getNextLocation(q, position, action)
    push!(path, position)
  end
  path
end

# Initialise the QLearning
q = readdlm("rewards.csv", ',') |> QLearning

# Training part
ϵ = 0.7
for _ in 1:5_000
  position = getStartingLocation(q)
  while !isterminal(position, q)
    action = getNextAction(q, position, ϵ)
    nextPostion = getNextLocation(q, position, action)
    update_qvalues!(q, position, nextPostion, action)
    position = nextPostion
  end
end

# Tests
startingPoint = [State(4,10), State(6,1), State(10,6), State(6,3)]

foreach(startingPoint) do point
  path = getShortestPath(q, point)
  grid = zeros(Int8, size(q.rewards))
  grid[path] .= 99
  display(grid)
  println('\n')
end