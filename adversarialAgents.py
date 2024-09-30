"""
CS311 Programming Assignment 2: Adversarial Search

Full Name: Frank Bautista

Brief description of my evaluation function:

TODO Briefly describe your evaluation function and why it improves the win rate
"""

import math, random, typing

import util
from game import Agent, Directions
from pacman import GameState



class ReflexAgent(Agent):
    """
    A reflex agent chooses the best action at each choice point by examining its alternatives via a state evaluation
    function.

    The code below is provided as a guide. You are welcome to change it as long as you don't modify the method
    signatures.
    """

    def __init__(self):
        self.lastPositions = []
        self.dc = None

    def getAction(self, gameState: GameState) -> str:
        """Choose the best action according to an evaluation function.

        Review pacman.py for the available methods on GameState.

        Args:
            gameState (GameState): Current game state

        Returns:
            str: Chosen legal action in this state
        """
        # Collect legal moves
        legalMoves = gameState.getLegalActions()

        # Compute the score for the successor states, choosing the highest scoring action
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        # Break ties randomly
        chosenIndex = random.choice(bestIndices)

        return legalMoves[chosenIndex]

    def evaluationFunction(self, gameState: GameState, action: str):
        """Compute score for current game state and proposed action"""
        successorGameState = gameState.generatePacmanSuccessor(action)
        return successorGameState.getScore()


def scoreEvaluationFunction(gameState: GameState) -> float:
    """
    Return score of gameState (as shown in Pac-Man GUI)

    This is the default evaluation function for adversarial search agents (not reflex agents)
    """
    return gameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    Abstract Base Class for Minimax, AlphaBeta and Expectimax agents.

    You do not need to modify this class, but it can be a helpful place to add attributes or methods that used by
    all your agents. Do not remove any existing functionality.
    """

    def __init__(self, evalFn=scoreEvaluationFunction, depth=1):
        self.index = 0  # Pac-Man is always agent index 0
        self.evaluationFunction = globals()[evalFn] if isinstance(evalFn, str) else evalFn
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """Minimax Agent"""


    def getAction(self, gameState: GameState) -> str:
        """Return the minimax action from the current gameState.

        * Use self.depth (depth limit) and self.evaluationFunction.
        * A "terminal" state is when Pac-Man won, Pac-Man lost or there are no legal moves.
        """

        """
        Some potentially useful methods on GameState (recall Pac-Man always has an agent index of 0, the ghosts >= 1):

        getLegalActions(agentIndex): Returns a list of legal actions for an agent
        generateSuccessor(agentIndex, action): Returns the successor game state after an agent takes an action
        getNumAgents(): Return the total number of agents in the game
        getScore(): Return the score corresponding to the current state of the game
        isWin(): Return True if GameState is a winning state
        gameState.isLose(): Return True if GameState is a losing state

        Idea: 
            - Depth limited search DFS with depth limited to 2 moves means we can only calculate our best move based on:
                - Ghosts optimal move (for themselves)
                - Pacman's surroundings (food, capsules, ghosts, etc.) to get to a winning state (GOAL)
            
                Ultimately we should be picking the best move based on our self.evaluationFunction which will rank the best next moves.
                We wont necessarily be making those choices ourselves, we only try to better account for those unaccounted variables while taking into account
                a bit of the future before giving them over to our evaluation function. Which will make the final choice. (we will of course randomly get to choose tie breakers)

            Things to keep in mind:
                It's unknown how many ghosts there will be so we will have to account for n ghosts.
                The height of our tree will be d(n+1) where d is the depth and n is the number of ghosts+1 (pacman)
        """

        maxHeight = self.depth*(gameState.getNumAgents() + 1)
        pacmanLegalActions = gameState.getLegalActions(0)
        bestScore = float("-inf")
        bestMoves = []

        for move in pacmanLegalActions:
            successorGameState = gameState.generateSuccessor(0, move)
            score = self.minimax(successorGameState, 1, 1, maxHeight)
            if score > bestScore:
                bestScore = score
                bestMoves = [move]
            elif score == bestScore:
                bestMoves.append(move)
        
        return random.choice(bestMoves)



            
    
    def minimax (self, gameState: GameState, agentIndex: int, height: int, maxHeight: int) -> int:
        if gameState.isWin() or gameState.isLose() or height >= maxHeight:
            return self.evaluationFunction(gameState)

        agentGameStates= [gameState.generateSuccessor(agentIndex, move) for move in gameState.getLegalActions(agentIndex)]  
        
        if agentIndex == gameState.getNumAgents()-1: #restart the cycle of agents signifies a new depth
            return max([self.minimax(state, 0, height+1, maxHeight) for state in agentGameStates])
        elif agentIndex == 0:
            return max([self.minimax(state, agentIndex+1, height+1, maxHeight) for state in agentGameStates])
        else:
            return min([self.minimax(state, agentIndex+1, height+1, maxHeight) for state in agentGameStates])
        


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Minimax agent with alpha-beta pruning
    """


    def getAction(self, gameState: GameState) -> str:
        """Return the minimax action with alpha-beta pruning from the current gameState.

        * Use self.depth (depth limit) and self.evaluationFunction.
        * A "terminal" state is when Pac-Man won, Pac-Man lost or there are no legal moves.
        """
        # TODO: Implement your Minimax Agent with alpha-beta pruning
        raise NotImplementedError()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Expectimax agent
    """


    def getAction(self, gameState):
        """Return the expectimax action from the current gameState.

        All ghosts should be modeled as choosing uniformly at random from their legal moves.

        * Use self.depth (depth limit) and self.evaluationFunction.
        * A "terminal" state is when Pac-Man won, Pac-Man lost or there are no legal moves.
        """
        # TODO: Implement your Expectimax Agent
        raise NotImplementedError()


def betterEvaluationFunction(gameState: GameState) -> float:
    """
    Return score of gameState using custom evaluation function that improves agent performance.
    """

    """
    The evaluation function takes the current GameStates (pacman.py) and returns a number,
    where higher numbers are better.

    Some methods/functions that may be useful for extracting game state:
    gameState.getPacmanPosition() # Pac-Man position
    gameState.getGhostPositions() # List of ghost positions
    gameState.getFood().asList() # List of positions of current food
    gameState.getCapsules() # List of positions of current capsules
    gameState.getGhostStates() # List of ghost states, including if current scared (via scaredTimer)
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    """

    # TODO: Implement your evaluation function
    raise Exception("Not implemented yet")


# Create short name for custom evaluation function
better = betterEvaluationFunction
