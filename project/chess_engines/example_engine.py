#!/usr/bin/python3
from project.chess_agents.agent import Agent
from project.chess_engines.uci_engine import UciEngine
import chess
from project.chess_agents.example_agent import ExampleAgent
from project.chess_utilities.example_utility import ExampleUtility
from project.chess_utilities.utility import Utility

if __name__ == "__main__":
    # Create your utility
    utility = Utility()
    # Create your agent
    agent = Agent(utility, 5.0)
    # Create the engine
    engine = UciEngine("Example engine", "Arne", agent)
    # Run the engine (will loop until the game is done or exited)
    engine.engine_operation()
