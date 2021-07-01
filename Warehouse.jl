using DelimitedFiles: readdlm_auto
using DelimitedFiles

# Map to be trained
rewards = readdlm("grid.csv", ',')

# Possible actions the agent can execute on the map
@enum Action up=1 right down left

q_table = let (rows, cols) = size(rewards)
  zeros(Float64, rows, cols, Action.size)
end

# Determine if the current state is terminal or if the agent can keep moving
isterminal(row, column) = rewards[row, column] != -1.0

# Starting from a random location
getStartingLocation() = let validPoints = findall(s -> s == -1, rewards)
  rand(validPoints) |> Tuple
end

getNextAction(row, column, ϵ) = rand() < ϵ ? argmax(q_table[row,column,:]) : rand(1:Action.size)

function getNextLocation(row, column, action_index)
  action = Action(action_index)
  move = if action == up;    (-1,0)
     elseif action == right; (0,1)
     elseif action == down;  (1,0)
     elseif action == left;  (0,-1)
     end
  next = (row, column) .+ move
  try
    checkbounds(q_table[:,:,1], next...)
    next
  catch e
    (row, column)
  end
end

function getShortestPath(row, column)
  path = [(row, column)]
  while !isterminal(row, column)
    action = getNextAction(row, column, 1.0)
    (row, column) = getNextLocation(row, column, action)
    push!(path, (row,column))
  end
  path
end

# Training part
ϵ = 0.7
discount = 0.9
learningRate = 0.9

for _ in 1:1000
  row, column = getStartingLocation()
  while !isterminal(row, column)
    action = getNextAction(row, column, ϵ)
    nextRow, nextColumn = getNextLocation(row, column, action)

    nextStepReward = rewards[nextRow,nextColumn]
    currentValue = q_table[row,column,action]
    temporal_difference = nextStepReward + discount * maximum(q_table[nextRow,nextColumn,:]) - currentValue

    newValue = currentValue + learningRate * temporal_difference
    q_table[row,column,action] = newValue

    row, column = (nextRow, nextColumn)
  end
end

getShortestPath(4,10) |> println
getShortestPath(6,1) |> println
getShortestPath(10,6) |> println
getShortestPath(6,3) |> println
