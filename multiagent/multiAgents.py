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

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

    def evaluationFunction(self, currentGameState, action):
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
        #The food - In the successor state, if you eat food, you are good
        #Ghosts - In you close to ghost and its scared, good
        # else, close and not scared, very bad

        prevFoodCount = currentGameState.getNumFood()
        newFoodCount = successorGameState.getNumFood()
        foodScore = prevFoodCount - newFoodCount     # eat food, good
        ghostList = successorGameState.getGhostPositions()
        ghostScore = 0

        foodPosList = newFood.asList()   #a list of tuples whose values are true
        closeFoodList = [manhattanDistance(food, newPos) for food in foodPosList]
        closestFood = 0
        if len(closeFoodList)>0:
            closestFood = min(closeFoodList)
        #closestFood should be smaller the better

        for i in range(len(ghostList)):
            ghostPos = ghostList[i]
            distance = manhattanDistance(ghostPos,newPos)
            isScared = newScaredTimes[i]>0
            if distance<2 and isScared:
                ghostScore += 100
            elif distance<2 and not isScared:
                return -1000
        return 100*foodScore + ghostScore + successorGameState.getScore() - closestFood


def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
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
        """
        "*** YOUR CODE HERE ***"
        
        def maxval(state,depth):
          if state.isWin() or state.isLose() or depth<1 : #depth 0 means hey dude we got where you wanna extend
            return self.evaluationFunction(state)
          maxValue = float("-inf")
          pacActionList = state.getLegalActions(0)
          for action in pacActionList:
            nextState = state.generateSuccessor(0,action)
            value = minval(nextState,depth,1)
            if value > maxValue :
              maxValue = value
          return maxValue
              
        def minval(state,depth,agent):
          # if terminal return utility
          if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
          minValue = float("inf") 
          ghostActionsList = state.getLegalActions(agent)
          for action in ghostActionsList:
            nextState = state.generateSuccessor(agent,action)
            if agent == gameState.getNumAgents()-1:               #we reached the last ghost agent
              value = maxval(nextState,depth-1)
            else:
              value = minval(nextState,depth,agent+1)
            if value < minValue :
              minValue = value
          return minValue
              
        res = []
        for action in gameState.getLegalActions(0):
          successorState = gameState.generateSuccessor(0,action)
          value = minval(successorState,self.depth,1)
          res.append((value,action))
        return max(res, key=lambda x: x[0])[1]
                 
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def ABmax(state,a,b,depth):
            if state.isWin() or state.isLose() or depth<1:
                return self.evaluationFunction(state)
            value = float("-inf")
            actionList = state.getLegalActions(0)  #agent 0 is the only guy who like to max
            for action in actionList:
                nextState = state.generateSuccessor(0,action)
                nextValue = ABmin(nextState,a,b,depth,1)
                if nextValue > value:
                    value = nextValue
                    if value > b:
                        return value
                    else:
                        a = value if value > a else a
            return value

        def ABmin(state,a,b,depth,agent):
            if state.isLose() or state.isWin():
                return self.evaluationFunction(state)
            value = float("inf")
            actionList = state.getLegalActions(agent)
            for action in actionList:
                nextState = state.generateSuccessor(agent,action)
                if agent == gameState.getNumAgents()-1:
                    nextValue = ABmax(nextState,a,b,depth-1)
                else:
                    nextValue = ABmin(nextState,a,b,depth,agent+1)
                if (nextValue < value):
                    value = nextValue
                if value < a:
                    return value
                else:
                    b = value if value < b else b
            return value

        alpha = float("-inf")
        beta = float("inf")
        bestAction = None
        maxV = float("-inf")     #the first move from root, we need max
        actions = gameState.getLegalActions(0)
        for action in actions:
            successorState = gameState.generateSuccessor(0,action)
            nextValue = ABmin(successorState,alpha,beta,self.depth,1)
            if nextValue > maxV:
                maxV = nextValue
                bestAction = action
            alpha = max(alpha,maxV)
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

        def expVal(state,depth,agent):
            if state.isLose() or state.isWin():
                return self.evaluationFunction(state)
            value = 0
            actions = state.getLegalActions(agent)
            if len(actions)==0:
                return self.evaluationFunction(state)
            #probability = 1.0/len(actions)
            for action in actions:
                nextState = state.generateSuccessor(agent,action)
                if agent+1 != state.getNumAgents():
                    value += expVal(nextState,depth,agent+1)
                else: #we done with ghosts
                    value += maxVal(nextState,depth-1)
            return value * 1.0 / len(actions)

        def maxVal(state, depth):
            if state.isLose() or state.isWin() or depth < 1:
                return self.evaluationFunction(state)
            actions = state.getLegalActions(0)
            value = float("-inf")
            for action in actions:
                nextState = state.generateSuccessor(0,action)
                nextValue = expVal(nextState, depth, 1)
                if nextValue > value:
                    value = nextValue
            return value

        if gameState.isLose() or gameState.isWin() or self.depth < 1:
            return self.evaluationFunction(gameState)
        actionList = gameState.getLegalActions(0)
        bestAction = None
        maxV = float("-inf")
        for action in actionList:
            successorState = gameState.generateSuccessor(0, action)
            nextValue = expVal(successorState, self.depth, 1)
            if maxV < nextValue:
                maxV = nextValue
                bestAction = action
        return bestAction

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).
      DESCRIPTION: Since Q5 using expectimax agent upon evaluationFunction(ghost act randomly), 
      It is reasonable to assume that the ghost acts randomly :)
      Also, the action of pacman looked like that the expectimax agent has been limited to a given depth - important info.
      If we do not consider the depth of the searching tree effect - an action queue of (stop, west) would be treat as the same to (west, stop)
      Thus we cannot stop pacman from stay on one state -- waiting ghost to chase it so that it can move :) WTF
      The way to solve this problem of "Un-preference" -- is to intro some add-in credit to let pacman move.
      First, we aboslutely need to consider the ghost coz eating the scared ghost would get 200 points of reward. 
      Always avoiding from meet with ghost leads usually lead to a score < 1000
      Then capsule would be appreciate while a ghost is getting near to pacman and if the ghost is not scared, 
      having capsules uneaten is bad, so info of capsule is introduced. 
      If the ghost is scared, being close to it is good, and there is no extra bonus (beyond the gameState.getScore() bump for eating a
      food) for being close to a food. In fact, discounting the game state food-eating bump/distance travelled decrement is 
      a good idea so that pacman is sufficiently motivated to chase after the ghost :)
      
    """
    "*** YOUR CODE HERE ***"

    if currentGameState.isWin() or currentGameState.isLose():
        return currentGameState.getScore()

    " Variables List"
    position = currentGameState.getPacmanPosition()
    
    capsules = currentGameState.getCapsules()
    numCapsules = len(capsules)
    
    walls = currentGameState.getWalls()
    
    ghostStates = currentGameState.getGhostStates()
    ghostDistance = []
    
    foodList = currentGameState.getFood().asList()
    foodDist, numFood = nearestItem(position, foodList, walls)
    
    capsuleDistance = 0
    capsuleCounter = 0
    
    nearestGhostDistance = 0
    ghostDistFeature = 0
    
    foodDistance = 1.0 / foodDist
    score = currentGameState.getScore()

    " Feature Weight List "
    capsuleDistWeight = 3
    capsuleCountWeight = 20
    ghostDistWeight = 40
    foodDistWeight = 0.25
    scoreWeight = 1.0

    for ghost in ghostStates:
        ghostDistance.append(((int((ghost.getPosition()[0])),int((ghost.getPosition()[1]))), 
                                manhattanDistance(position, ghost.getPosition()), ghost.scaredTimer)
                            )

    if len(ghostDistance) > 0:
        nearestGhost = min(ghostDistance, key=itemgetter(1))
        nearestGhostDistance = nearestGhost[1]
    else: 
        nearestGhostDistance = 100000

    if nearestGhost[2]:
        ghostDistFeature = 1.0 / nearestGhostDistance
        foodDistWeight = 0
        scoreWeight = 0.99
    elif numCapsules:
        capsuleCounter = -1
        capsuleDist, _ = nearestItem(position, capsules, walls)
        capsuleDistance = 1.0 / capsuleDist

    utilityScore = (scoreWeight * score +
                    capsuleDistWeight * capsuleDistance +
                    capsuleCountWeight * capsuleCounter +
                    ghostDistWeight * ghostDistFeature +
                    foodDistWeight * foodDistance)
    return utilityScore

def nearestItem(position, items, walls):
    if not items:
        return 0, None
    closed = set()
    fringe = util.Queue()
    fringe.push((position, 0))
    while not fringe.isEmpty():
        (x, y), cost = fringe.pop()
        if (x, y) in items:
            return cost, (x, y)
        if (x, y) in closed:
            continue
        closed.add((x, y))
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for (nx, ny) in [(x+dx, y+dy) for dx, dy in directions if not walls[x+dx][y+dy]]:
            fringe.push(((nx, ny), cost+1))
    return 0, None

# Abbreviation
better = betterEvaluationFunction

# Abbreviation
better = betterEvaluationFunction

