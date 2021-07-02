using DelimitedFiles

# Map to be trained
rewards = readdlm("grid.csv", ',')

# Possible actions the agent can execute on the map
@enum Action up=1 right down left

q_table = let (rows, cols) = size(rewards)
  zeros(Float64, rows, cols, Action.size)
end

# Determine if the current state is terminal or if the agent can keep moving
isterminal(position::CartesianIndex) = rewards[position] != -1.0

# Starting from a random location
getStartingLocation() = let validPoints = findall(s -> s == -1, rewards)
  rand(validPoints)
end

getNextAction(position::CartesianIndex, ϵ) = rand() < ϵ ? argmax(q_table[position,:]) : rand(1:Action.size)

function getNextLocation(position::CartesianIndex, action_index)
  action = Action(action_index)
  move = if action == up;    CartesianIndex(-1,0)
     elseif action == right; CartesianIndex(0,1)
     elseif action == down;  CartesianIndex(1,0)
     elseif action == left;  CartesianIndex(0,-1)
     end
  next = position + move
  try
    checkbounds(q_table[:,:,1], next)
    next
  catch _
    position
  end
end

function getShortestPath(position::CartesianIndex)
  path = [position]
  while !isterminal(position)
    action = getNextAction(position, 1.0)
    position = getNextLocation(position, action)
    push!(path, position)
  end
  path
end

# Training part
ϵ = 0.7
discount = 0.9
learningRate = 0.9

for _ in 1:1000
  position = getStartingLocation()
  while !isterminal(position)
    action = getNextAction(position, ϵ)
    nextPostion = getNextLocation(position, action)

    nextStepReward = rewards[nextPostion]
    currentValue = q_table[position,action]
    temporal_difference = nextStepReward + discount * maximum(q_table[nextPostion,:]) - currentValue

    newValue = currentValue + learningRate * temporal_difference
    q_table[position,action] = newValue

    position = nextPostion
  end
end

function showpath(path::Vector{CartesianIndex{2}})
  grid = ones(Int8, size(rewards))
  grid[path] .= 0
  grid
end

CartesianIndex(4,10) |> getShortestPath |> showpath |> display
println()
CartesianIndex(6,1)  |> getShortestPath |> showpath |> display
println()
CartesianIndex(10,6) |> getShortestPath |> showpath |> display
println()
CartesianIndex(6,3)  |> getShortestPath |> showpath |> display
