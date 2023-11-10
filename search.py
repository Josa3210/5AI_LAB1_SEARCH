# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""
from typing import List, Tuple

import util
from game import Directions

Coordinate = tuple[int, int] # (x, y)
State = tuple[Coordinate, str, int] # (coordinate, direction, cost)
StateWithParent = tuple[Coordinate, str, int, "StateWithParent"]

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem: SearchProblem) -> list[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    state : State = (problem.getStartState(),'',0)
    moves : list[Directions]  = dfsHelper(state, problem, set())[0]
    moves.pop() # Last move is always gibberish
    moves.reverse() # Moves are returned in reverse order
    return moves

def breadthFirstSearch(problem) -> list[Directions]:
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    queue : util.Queue = util.Queue() # We use a queue to enqueue al states from the current depth.
    exploredStates : set[Coordinate] = set() # We use this to make sure we do not re-explore states
    state : StateWithParent = (problem.getStartState(),'',0, None) # The state with parent variable ensures we can find our way from the solution to the origin without using recursion
    resultState : StateWithParent = None # When a goalState is found this will be set with that goalState.
    queue.push(state) # We start our search by pushing the startState as the first state to check.
    while (not queue.isEmpty()): 
        state = queue.pop() # We examine the state that was pushed the least recently on the queue
        exploredStates.add(state[0]) # We add this state to our exploredStates.
        if (problem.isGoalState(state[0])): # If we have arrived at our goal state, set the resultState variable and break out of the loop.
            resultState = state
            break
        for newState in problem.getSuccessors(state[0]): # Otherwise we loop over all possible states that can be reached from the currentState.
            if (not newState[0] in exploredStates): # If the state is unexplored we enqueue for our bfs.
                newStateWithParent = (newState[0], newState[1], newState[2], state) # we add the currentState as the parent of the new state. this way we can follow the propagation of the algorithm.
                queue.push(newStateWithParent)
    moves = stateWithParentToDirections(resultState) # this function allows us to back-track a certain position to our origin and write the moves to a list.
    return moves

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch




def dfsHelper(state : State, problem, exploredStates : set[Coordinate]) -> Tuple[list[Directions], set[Coordinate]]:
    exploredStates.add(state[0]) # add the current state to the explored state. This ensures that we don't pass through it twice.
    if (problem.isGoalState(state[0])): #if we have reached the goal state return the Direction that is held within the state variable.
        return [directionMapper(state[1])], set() # we pass an empty set because that is more memory friendly
    for newState in problem.getSuccessors(state[0]):
        if (newState[0] in exploredStates): # newState is also a State object, the explored States only holds coordinates.
            continue # if a generated state is already explored we ignore it.
        moveList, exploredStates = dfsHelper(newState, problem, exploredStates) # if an unexplored state is encountered we will check it now.
        if (moveList is not None): # if None was returned, this means that state search has not found any viable paths. Otherwise we have successfully found a solution
            moveList.append(directionMapper(state[1])) # list.appends returns None, so we need to appends separately, we append the Direction from the current state variable
            return moveList, set() 
    return None, exploredStates


def directionMapper(directionString : str) -> Directions:
    direction: Directions
    if (directionString == "South"):
        direction = Directions.SOUTH
    elif (directionString == "West"):
        direction = Directions.WEST
    elif (directionString == "North"):
        direction = Directions.NORTH
    else:
        direction = Directions.EAST
    return direction

def stateWithParentToDirections(state : StateWithParent) -> list[Directions]:
    """Given a certain state that has parent information. back-track through the tree to the origin and record all moves to a list.

    Args:
        state (StateWithParent): a state that holds a parent that is also a StateWithParent.

    Returns:
        list[Directions]: The direction in correct order FROM origin TO solution.
    """
    parent : StateWithParent = state[3]
    currentState : StateWithParent = state
    moves : list[Directions] = []
    while (parent is not None): # as long as there is parent information we will keep the loop going.
        if (currentState[1] in ["North", "East", "West", "South"]): #If the direction is gibberish we won't consider it for recording. 
            moves.insert(0, directionMapper(currentState[1])) #insert the move at the beginning of the list.
        currentState = parent # the next state we will check will be our current parent.
        parent = currentState[3] # the new nex parent will be the parent of the new next state.
    return moves