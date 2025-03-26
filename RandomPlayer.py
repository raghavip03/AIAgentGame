''' RandomPlayer.py
A player for the game of K-in-a-Row (on N by M board with forbidden squares.)

'''



from agent_base import KAgent
import game_types
from random import randint
GAME_TYPE = None

class OurAgent(KAgent):

    def __init__(self, twin=False):
        self.twin = twin
        self.nickname = 'Randy'
        if twin: self.nickname = "Randy-Junior"
        self.long_name = 'Random Walker'
        if twin: self.long_name = "Random Walker Junior"
        self.my_past_utterances = None
        self.opponent_past_utterances = None
        self.repeat_count = None
        self.utt_count = None

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
       self.who_i_play = what_side_to_play
       self.opponent_nickname = opponent_nickname
       self.time_limit = expected_time_per_move
       global GAME_TYPE
       GAME_TYPE = game_type
       print("Oh, I love playing randomly at ", game_type.long_name)
       self.my_past_utterances = []
       self.opponent_past_utterances = []
       self.repeat_count = 0
       self.utt_count = 0
       if self.twin: self.utt_count = 5 # Offset the twin's utterances.

       return "OK"

    def introduce(self):
        if self.twin:
            remark = "Call me the Junior Random Walker."
        else:
            remark = "My name is "+self.long_name+". Or is it Walky Rander?"
        return remark

    def nickname(self): return self.nickname

    def makeMove(self, state, lastUtterance, timeLimit):
        possibleMoves = successors_and_moves(state)
        myMove = chooseMove(possibleMoves)
        myUtterance = self.nextUtterance()
        newState, newMove = myMove
        return [[newMove, newState], myUtterance]

    def nextUtterance(self):
        if self.repeat_count > 1: return "I am randomed out now."
        n = len(UTTERANCE_BANK)
        if self.utt_count == n:
            self.utt_count = 0
            self.repeat_count += 1
        this_utterance = UTTERANCE_BANK[self.utt_count]
        self.utt_count += 1
        return this_utterance

# OPTIONAL THINGS TO KEEP TRACK OF:

#  WHO_MY_OPPONENT_PLAYS = other(WHO_I_PLAY)
#  MY_PAST_UTTERANCES = []
#  OPPONENT_PAST_UTTERANCES = []
#  UTTERANCE_COUNT = 0
#  REPEAT_COUNT = 0 or a table of these if you are reusing different utterances


def other(p):
    if p=='X': return 'O'
    return 'X'

def chooseMove(statesAndMoves):
    states, moves = statesAndMoves
    if states==[]: return None
    random_index = randint(0, len(states)-1)
    my_choice = [states[random_index], moves[random_index]]
    return my_choice

def successors_and_moves(state):
    b = state.board
    p = state.whose_move
    o = other(p)
    new_states = []
    moves = []
    mCols = len(b[0])
    nRows = len(b)

    for i in range(nRows):
        for j in range(mCols):
            if b[i][j] != ' ': continue
            news = do_move(state, i, j, o)
            new_states.append(news)
            moves.append([i, j])
    return [new_states, moves]

def do_move(state, i, j, o):
            news = game_types.State(old=state)
            news.board[i][j] = state.whose_move
            news.whose_move = o
            return news

UTTERANCE_BANK = ["How's that for random?",
                  "Flip!",
                  "Spin!",
                  "I hope this is my lucky day!",
                  "How's this move for high noise to signal ratio?",
                  "Uniformly distributed. That's me.",
                  "Maybe I'll look into Bayes' Nets in the future.",
                  "Eenie Meenie Miney Mo.  I hope I'm getting K in a row.",
                  "Your choice is probably more informed than mine.",
                  "If I only had a brain.",
                  "I'd while away the hours, playing K in a Row.",
                  "So much fun.",
                  "Roll the dice!",
                  "Yes, I am on a roll -- of my virtual dice.",
                  "randint is my cousin.",
                  "I like to spread my influence around on the board."]


def test():
    global GAME_TYPE
    GAME_TYPE = game_types.TTT
    print(GAME_TYPE)
    h = OurAgent()
    print("I am ", h.nickname)

    ttt = GAME_TYPE.initial_state
    print("ttt initial state: ")
    print(ttt)
    print("successors_and_moves: ")
    print(successors_and_moves(ttt))

if __name__=="__main__":
    test()

