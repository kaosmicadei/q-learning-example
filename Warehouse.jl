using DelimitedFiles

# Possible actions the agent can execute.
@enum Action up=1 right down left

# Define an enviroment.
struct Environment
  rewards::Matrix{Float64}
  qTable::Array{Float64,3}
end

Environment(rewards) = let 
  (rows, cols) = size(rewards)
  qTable = zeros(Float64, rows, cols, Action.size)
  Environment(rewards, qTable)
end

function update_qTable!(e::Environment, currentPosition::CartesianIndex, nextPosition::CartesianIndex, action::Action; α=0.9, γ=0.9)
  # α : learning rate
  # γ : discount factor
  action_index = Int(action)
  reward = e.rewards[nextPosition]
  qValue = e.qTable[currentPosition, action_index]
  optimalNextAction = maximum(e.qTable[nextPosition,:])
  newValue = reward + γ * optimalNextAction
  temporalDifference = newValue - qValue
  e.qTable[currentPosition, action_index] = qValue + α * temporalDifference
end

# Determine if the current state is terminal or if the agent can keep moving
isterminal(position::CartesianIndex, e::Environment) = e.rewards[position] != -1.0

# Starting from a random location
getStartingLocation(e::Environment) = let validPoints = findall(s -> s == -1, e.rewards)
  rand(validPoints)
end

getNextAction(e::Environment, position::CartesianIndex, ϵ) = let
  index = rand() < ϵ ? argmax(e.qTable[position,:]) : rand(1:Action.size)
  Action(index)
end

function getNextLocation(e::Environment, position::CartesianIndex, action::Action)
  move = if action == up;    CartesianIndex(-1,0)
     elseif action == right; CartesianIndex(0,1)
     elseif action == down;  CartesianIndex(1,0)
     elseif action == left;  CartesianIndex(0,-1)
     end
  next = position + move
  try
    checkbounds(e.qTable[:,:,1], next)
    next
  catch _
    position
  end
end

function getShortestPath(e::Environment, position::CartesianIndex)
  path = [position]
  while !isterminal(position, e)
    action = getNextAction(e, position, 1.0)
    position = getNextLocation(e, position, action)
    push!(path, position)
  end
  path
end

# Initialise the environment
env = readdlm("rewards.csv", ',') |> Environment

# Training part
ϵ = 0.7
for _ in 1:5_000
  position = getStartingLocation(env)
  while !isterminal(position, env)
    action = getNextAction(env, position, ϵ)
    nextPostion = getNextLocation(env, position, action)
    update_qTable!(env, position, nextPostion, action)
    position = nextPostion
  end
end

# Tests
startingPoint = [CartesianIndex(4,10), CartesianIndex(6,1), CartesianIndex(10,6), CartesianIndex(6,3)]

foreach(startingPoint) do point 
  path = getShortestPath(env, point)
  grid = zeros(Int8, size(env.rewards))
  grid[path] .= 99
  display(grid)
  println('\n')
end