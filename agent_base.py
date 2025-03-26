'''
agent_base.py

Base class to be subclassed to create an agent for playing
"K-in-a-Row with Forbidden Squares" and related games.

CSE 473, University of Washington

THIS IS A TEMPLATE WITH STUBS FOR THE REQUIRED FUNCTIONS.

IMPORT IT INTO YOUR OWN AGENT MODULE AND SUBCLASS KAgent.
OVERRIDE METHODS AS NEEDED TO CREATE YOUR OWN AGENT.

YOU CAN PUT INTO YOUR MODULE WHATEVER ADDITIONAL FUNCTIONS
YOU NEED IN ORDER TO ACHIEVE YOUR AGENT IMPLEMENTATION.

'''


AUTHORS = 'Jane Smith and Laura Lee' # Override this in your agent file.

import time

# Base class for all K-in-a-Row agents.

class KAgent:

    def __init__(self, twin=False):
        self.twin=False
        self.nickname = 'Nic'
        if twin: self.nickname += '2'
        self.long_name = 'Templatus Skeletus'
        if twin: self.long_name += ' II'
        self.persona = 'bland'
        self.voice_info = {'Chrome': 10, 'Firefox': 2, 'other': 0}
        self.playing = "don't know yet" # e.g., "X" or "O".
        self.image = None

    def introduce():
        intro = '\nMy name is Templatus Skeletus.\n'+\
            '"An instructor" made me.\n'+\
            'Somebody please turn me into a real game-playing agent!\n'
        return intro

    def nickname():
        return self.nickname

    # Receive and acknowledge information about the game from
    # the game master:
    def prepare(
            self,
            game_type,
            what_side_to_play,
            opponent_nickname,
            expected_time_per_move = 0.1, # Time limits can be
                                          # changed mid-game by the game master.
            utterances_matter = True):    # If False, just return 'OK' for each utterance.

       # Write code to save the relevant information in variables
       # local to this instance of the agent.
       # Game-type info can be in global variables.
       print("Change this to return 'OK' when ready to test the method.")
       return "Not-OK"

    def makeMove(currentState, currentRemark, timeLimit=10000):
        print("makeMove has been called")

        print("code to compute a good move should go here.")
        # Here's a placeholder:
        a_default_move = [0, 0] # This might be legal ONCE in a game,
        # if the square is not forbidden or already occupied.

        newState = currentState # This is not allowed, and even if
        # it were allowed, the newState should be a deep COPY of the old.

        newRemark = "I need to think of something appropriate.\n" +\
        "Well, I guess I can say that this move is probably illegal."

        print("Returning from makeMove")
        return [[a_default_move, newState], newRemark]


    # The main adversarial search function:
    def minimax(
            state,
            depthRemaining,
            pruning=False,
            alpha=None,
            beta=None,
            zHashing=None):
        print("Calling minimax. We need to implement its body.")

        default_score = 0 # Value of the passed-in state. Needs to be computed.

        return [default_score, "my own optional stuff", "more of my stuff"]
        # Only the score is required here but other stuff can be returned
        # in the list, after the score, in case you want to pass info
        # back from recursive calls that might be used in your utterances,
        # etc.

    def staticEval(state):
        print('calling staticEval. Its value needs to be computed!')
        # Values should be higher when the states are better for X,
        # lower when better for O.
        return 0


GAME_TYPE = None  # not known yet.
