# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        "*** YOUR CODE HERE ***"

        """
        DESCRIPTION FOR OUR EVALUATION FUNCTION:
        This evaluation function considers several factors to make better decisions:
        1. Base score of the game state
        2. Distance to nearest food (closer is better, using reciprocal)
        3. Distance to ghosts (farther is better, unless ghosts are scared)
        4. Number of remaining food (less is better)
        5. Avoid stopping (STOP action is penalized)
        
        The function balances these factors to make Pacman prioritize:
        - Eating nearby food
        - Staying away from non-scared ghosts
        - Approaching scared ghosts for extra points
        - Making progress through the maze
        """
        
        # Start with the current score
        score = successorGameState.getScore()
        
        # Penalize the STOP action to encourage movement
        if action == Directions.STOP:
            score -= 10
            
        # Calculate distance to the nearest food
        foodList = newFood.asList()
        if len(foodList) > 0:  # If there's food left
            foodDistances = [manhattanDistance(newPos, food) for food in foodList]
            minFoodDistance = min(foodDistances)
            # Using reciprocal so closer food gives higher score
            score += 1.0 / (minFoodDistance + 1)  # Add 1 to avoid division by zero
            
            # Reward having less food left
            score += 100 / (len(foodList) + 1)
        
        # Handle ghosts - stay away from non-scared ghosts, go after scared ones
        for i, ghostState in enumerate(newGhostStates):
            ghostPos = ghostState.getPosition()
            ghostDistance = manhattanDistance(newPos, ghostPos)
            
            # If ghost is very close and not scared, heavily penalize
            if ghostDistance < 2 and ghostState.scaredTimer == 0:
                score -= 500
            # If ghost is somewhat close and not scared, penalize based on distance
            elif ghostDistance < 4 and ghostState.scaredTimer == 0:
                score -= 100 / (ghostDistance + 1)
            # If ghost is scared, give bonus for being close to it
            elif ghostState.scaredTimer > 0:
                score += 200 / (ghostDistance + 1)
        
        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        # Get the best action according to the minimax algorithm
        def minimax(state, agentIndex, depth):
            "*** YOUR CODE HERE ***"
            # Check if we've reached a terminal state or maximum depth
            if state.isWin() or state.isLose() or depth == self.depth * state.getNumAgents():
                return self.evaluationFunction(state), None
            
            # Get legal actions for current agent
            legalActions = state.getLegalActions(agentIndex)
            
            # If no legal actions, evaluate current state
            if not legalActions:
                return self.evaluationFunction(state), None
            
            # Determine the next agent index
            nextAgentIndex = (agentIndex + 1) % state.getNumAgents()
            
            # Increment depth when we cycle back to Pacman (agent 0)
            nextDepth = depth + 1
            
            # If it's Pacman's turn (MAX player)
            if agentIndex == 0:
                bestValue = float('-inf')
                bestAction = None
                
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value, _ = minimax(successor, nextAgentIndex, nextDepth)
                    
                    if value > bestValue:
                        bestValue = value
                        bestAction = action
                        
                return bestValue, bestAction
            
            # If it's a ghost's turn (MIN player)
            else:
                bestValue = float('inf')
                bestAction = None
                
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value, _ = minimax(successor, nextAgentIndex, nextDepth)
                    
                    if value < bestValue:
                        bestValue = value
                        bestAction = action
                        
                return bestValue, bestAction
        
        # Start minimax from Pacman's perspective (agent 0) at depth 0
        _, action = minimax(gameState, 0, 0)
        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def alphaBeta(state, agentIndex, depth, alpha, beta):
            "*** YOUR CODE HERE ***"
            # Terminal states: win, lose, or reached maximum depth
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state), None
                
            # If we've reached the maximum depth
            if depth == self.depth:
                return self.evaluationFunction(state), None
                
            # Get legal actions for current agent
            legalActions = state.getLegalActions(agentIndex)
            
            # Determine next agent index
            nextAgentIndex = (agentIndex + 1) % state.getNumAgents()
            
            # Update depth when we've processed all agents and are back to Pacman
            nextDepth = depth + 1 if nextAgentIndex == 0 else depth
            
            # If it's Pacman's turn (MAX player)
            if agentIndex == 0:
                bestValue = float('-inf')
                bestAction = None
                
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value, _ = alphaBeta(successor, nextAgentIndex, nextDepth, alpha, beta)
                    
                    # Update best value and action
                    if value > bestValue:
                        bestValue = value
                        bestAction = action
                    
                    # Update alpha (for max player)
                    if value > alpha:
                        alpha = value
                    
                    # Prune if we can (but not on equality)
                    if value > beta:
                        return value, action
                        
                return bestValue, bestAction
            
            # If it's a ghost's turn (MIN player)
            else:
                bestValue = float('inf')
                bestAction = None
                
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value, _ = alphaBeta(successor, nextAgentIndex, nextDepth, alpha, beta)
                    
                    # Update best value and action
                    if value < bestValue:
                        bestValue = value
                        bestAction = action
                    
                    # Update beta (for min player)
                    if value < beta:
                        beta = value
                    
                    # Prune if we can (but not on equality)
                    if value < alpha:
                        return value, action
                        
                return bestValue, bestAction
        
        # Start alpha-beta from Pacman's perspective (agent 0) at depth 0
        # Initialize alpha to negative infinity and beta to positive infinity
        _, action = alphaBeta(gameState, 0, 0, float('-inf'), float('inf'))
        return action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        def expectimax(state, agentIndex, depth):
            "*** YOUR CODE HERE ***"
            # Terminal states: win, lose, or reached maximum depth
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state), None
                
            # If we've reached the maximum depth
            if depth == self.depth:
                return self.evaluationFunction(state), None
                
            # Get legal actions for current agent
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state), None
            
            # Determine next agent index
            nextAgentIndex = (agentIndex + 1) % state.getNumAgents()
            
            # Update depth when we've processed all agents and are back to Pacman
            nextDepth = depth + 1 if nextAgentIndex == 0 else depth
            
            # If it's Pacman's turn (MAX player)
            if agentIndex == 0:
                bestValue = float('-inf')
                bestAction = None
                
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value, _ = expectimax(successor, nextAgentIndex, nextDepth)
                    
                    # Update best value and action
                    if value > bestValue:
                        bestValue = value
                        bestAction = action
                        
                return bestValue, bestAction
            
            # If it's a ghost's turn (EXPECTATION player)
            else:
                # Calculate expected value across all actions
                totalValue = 0
                # Uniform probability for each action
                probability = 1.0 / len(legalActions)
                
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value, _ = expectimax(successor, nextAgentIndex, nextDepth)
                    totalValue += probability * value
                
                # For ghosts, we return the expected value and None for action
                # since they don't need to choose a best action
                return totalValue, None
        
        # Start expectimax from Pacman's perspective (agent 0) at depth 0
        _, action = expectimax(gameState, 0, 0)
        return action

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: This evaluation function considers multiple factors:
    1. Current score - base value from the game
    2. Food distance - prioritizes closer food (using reciprocal distance)
    3. Food count - fewer remaining food items is better
    4. Ghost distance - prefers to stay away from non-scared ghosts
    5. Scared ghosts - actively hunts ghosts when they're scared
    6. Capsules - prioritizes getting power pellets when ghosts are nearby
    
    All these factors are weighted and combined to create a comprehensive
    evaluation that enables Pacman to make smart decisions about hunting ghosts,
    collecting food efficiently, and avoiding danger.
    """
    # Current position and score
    pacmanPosition = currentGameState.getPacmanPosition()
    currentScore = currentGameState.getScore()
    
    # Food information
    foodGrid = currentGameState.getFood()
    foodList = foodGrid.asList()
    foodCount = len(foodList)
    
    # Ghost information
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    
    # Capsule information
    capsules = currentGameState.getCapsules()
    capsuleCount = len(capsules)
    
    # Initialize the evaluation score with current game score
    score = currentScore
    
    # Handle food - prioritize closer food
    if foodCount > 0:
        foodDistances = [manhattanDistance(pacmanPosition, food) for food in foodList]
        minFoodDistance = min(foodDistances)
        # Add inverse of distance to closest food (closer is better)
        score += 10.0 / (minFoodDistance + 1)
        
        # Penalize having more food left (encourages eating)
        score -= 4 * foodCount
    
    # Handle ghosts - stay away from active ghosts, hunt scared ones
    for i, ghostState in enumerate(ghostStates):
        ghostPos = ghostState.getPosition()
        ghostDistance = manhattanDistance(pacmanPosition, ghostPos)
        
        # If ghost is scared, try to catch it
        if ghostState.scaredTimer > 0:
            # If scared ghost is close, go for it
            if ghostDistance < ghostState.scaredTimer:
                score += 200.0 / (ghostDistance + 1)
            # If we can't reach it in time, don't bother
        else:
            # Stay away from active ghosts
            if ghostDistance < 3:
                # Extreme penalty for being very close to a ghost
                score -= 500.0 / (ghostDistance + 0.1)
            elif ghostDistance < 5:
                # Moderate penalty for being somewhat close
                score -= 100.0 / (ghostDistance + 0.1)
    
    # Handle capsules - prioritize them when ghosts are nearby
    if capsuleCount > 0:
        capsuleDistances = [manhattanDistance(pacmanPosition, capsule) for capsule in capsules]
        minCapsuleDistance = min(capsuleDistances)
        
        # Check if there are non-scared ghosts nearby
        dangerous_ghosts_nearby = any(ghostState.scaredTimer == 0 and 
                                    manhattanDistance(pacmanPosition, ghostState.getPosition()) < 5 
                                    for ghostState in ghostStates)
        
        # If dangerous ghosts are nearby, prioritize capsules
        if dangerous_ghosts_nearby:
            score += 100.0 / (minCapsuleDistance + 1)
        else:
            # Still good to get capsules, but less urgent
            score += 25.0 / (minCapsuleDistance + 1)
    
    return score

# Abbreviation
better = betterEvaluationFunction
