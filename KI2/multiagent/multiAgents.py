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
        
        score = successorGameState.getScore()
        
        if action == Directions.STOP:
            score -= 10
            
        foodList = newFood.asList()
        if len(foodList) > 0:
            foodDistances = [manhattanDistance(newPos, food) for food in foodList]
            minFoodDistance = min(foodDistances)
            score += 1.0 / (minFoodDistance + 1)
            
            score += 100 / (len(foodList) + 1)
        
        for i, ghostState in enumerate(newGhostStates):
            ghostPos = ghostState.getPosition()
            ghostDistance = manhattanDistance(newPos, ghostPos)
            
            if ghostDistance < 2 and ghostState.scaredTimer == 0:
                score -= 500
            elif ghostDistance < 4 and ghostState.scaredTimer == 0:
                score -= 100 / (ghostDistance + 1)
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
        def minimax(state, agentIndex, depth):
            "*** YOUR CODE HERE ***"
            if state.isWin() or state.isLose() or depth == self.depth * state.getNumAgents():
                return self.evaluationFunction(state), None
            
            legalActions = state.getLegalActions(agentIndex)
            
            if not legalActions:
                return self.evaluationFunction(state), None
            
            nextAgentIndex = (agentIndex + 1) % state.getNumAgents()
            
            nextDepth = depth + 1
            
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
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state), None
                
            if depth == self.depth:
                return self.evaluationFunction(state), None
                
            legalActions = state.getLegalActions(agentIndex)
            
            nextAgentIndex = (agentIndex + 1) % state.getNumAgents()
            
            nextDepth = depth + 1 if nextAgentIndex == 0 else depth
            
            if agentIndex == 0:
                bestValue = float('-inf')
                bestAction = None
                
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value, _ = alphaBeta(successor, nextAgentIndex, nextDepth, alpha, beta)
                    
                    if value > bestValue:
                        bestValue = value
                        bestAction = action
                    
                    if value > alpha:
                        alpha = value
                    
                    if value > beta:
                        return value, action
                        
                return bestValue, bestAction
            
            else:
                bestValue = float('inf')
                bestAction = None
                
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value, _ = alphaBeta(successor, nextAgentIndex, nextDepth, alpha, beta)
                    
                    if value < bestValue:
                        bestValue = value
                        bestAction = action
                    
                    if value < beta:
                        beta = value
                    
                    if value < alpha:
                        return value, action
                        
                return bestValue, bestAction
        
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
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state), None
                
            if depth == self.depth:
                return self.evaluationFunction(state), None
                
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state), None
            
            nextAgentIndex = (agentIndex + 1) % state.getNumAgents()
            
            nextDepth = depth + 1 if nextAgentIndex == 0 else depth
            
            if agentIndex == 0:
                bestValue = float('-inf')
                bestAction = None
                
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value, _ = expectimax(successor, nextAgentIndex, nextDepth)
                    
                    if value > bestValue:
                        bestValue = value
                        bestAction = action
                        
                return bestValue, bestAction
            
            else:
                totalValue = 0
                probability = 1.0 / len(legalActions)
                
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value, _ = expectimax(successor, nextAgentIndex, nextDepth)
                    totalValue += probability * value
                
                return totalValue, None
        
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
    pacmanPosition = currentGameState.getPacmanPosition()
    currentScore = currentGameState.getScore()
    
    foodGrid = currentGameState.getFood()
    foodList = foodGrid.asList()
    foodCount = len(foodList)
    
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    
    capsules = currentGameState.getCapsules()
    capsuleCount = len(capsules)
    
    score = currentScore
    
    if foodCount > 0:
        foodDistances = [manhattanDistance(pacmanPosition, food) for food in foodList]
        minFoodDistance = min(foodDistances)
        score += 10.0 / (minFoodDistance + 1)
        
        score -= 4 * foodCount
    
    for i, ghostState in enumerate(ghostStates):
        ghostPos = ghostState.getPosition()
        ghostDistance = manhattanDistance(pacmanPosition, ghostPos)
        
        if ghostState.scaredTimer > 0:
            if ghostDistance < ghostState.scaredTimer:
                score += 200.0 / (ghostDistance + 1)
        else:
            if ghostDistance < 3:
                score -= 500.0 / (ghostDistance + 0.1)
            elif ghostDistance < 5:
                score -= 100.0 / (ghostDistance + 0.1)
    
    if capsuleCount > 0:
        capsuleDistances = [manhattanDistance(pacmanPosition, capsule) for capsule in capsules]
        minCapsuleDistance = min(capsuleDistances)
        
        dangerous_ghosts_nearby = any(ghostState.scaredTimer == 0 and 
                                    manhattanDistance(pacmanPosition, ghostState.getPosition()) < 5 
                                    for ghostState in ghostStates)
        
        if dangerous_ghosts_nearby:
            score += 100.0 / (minCapsuleDistance + 1)
        else:
            score += 25.0 / (minCapsuleDistance + 1)
    
    return score

# Abbreviation
better = betterEvaluationFunction
