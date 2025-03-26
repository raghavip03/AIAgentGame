'''
<raghavip>_KInARow.py
Authors: <Putluri, Raghavi>
  Example:
    Authors: Putluri, Raghavi

An agent for playing "K-in-a-Row with Forbidden Squares" and related games.
CSE 473, University of Washington

THIS IS A TEMPLATE WITH STUBS FOR THE REQUIRED FUNCTIONS.
YOU CAN ADD WHATEVER ADDITIONAL FUNCTIONS YOU NEED IN ORDER
TO PROVIDE A GOOD STRUCTURE FOR YOUR IMPLEMENTATION.

'''

from agent_base import KAgent
from game_types import State, Game_Type, deep_copy, TTT_INITIAL_STATE_DATA, FIVE_INITIAL_STATE_DATA, CASSINI_INITIAL_STATE_DATA
AUTHORS = 'Raghavi Putluri'

import time # You'll probably need this to avoid losing a
 # game due to exceeding a time limit.

# Create your own type of agent by subclassing KAgent:

class OurAgent(KAgent):  # Keep the class name "OurAgent" so a game master
    # knows how to instantiate your agent class.

    def __init__(self, twin=False):
        self.twin=twin
        self.nickname = 'Goofy'
        if twin: self.nickname = 'Pluto'
        self.long_name = 'Goofy the Dog'
        if twin: self.long_name = 'Pluto the Pet'
        self.persona = 'sports commentator'
        self.voice_info = {'Chrome': 10, 'Firefox': 2, 'other': 0}
        self.playing = "X" # e.g., "X" or "O".

    def introduce(self):
        intro = completionGoofy('Introduce yourself, you are Goofy \n'+\
                                        'a funny player for K-in-a-row game \n'+\
                                            'in one sentence')
        if self.twin: intro = completionPluto('Introduce yourself, you are Pluto \n'+\
                                        'a confused and straight-forward player for K-in-a-row game \n'+\
                                            'in one sentence')
        return intro

    # Receive and acknowledge information about the game from
    # the game master:
    def prepare(
        self,
        game_type,
        what_side_to_play,
        opponent_nickname,
        expected_time_per_move = 0.1, # Time limits can be
                                      # changed mid-game by the game master.
        utterances_matter=True):      # If False, just return 'OK' for each utterance.

       # Write code to save the relevant information in variables
       # local to this instance of the agent.
       # Game-type info can be in global variables.
       #print("Change this to return 'OK' when ready to test the method.")
       self.game_type = game_type
       self.what_side_to_play = what_side_to_play
       self.opponent_nickname = opponent_nickname
       self.expected_time_per_move = expected_time_per_move
       self.utterances_matter = utterances_matter
       self.my_past_utterances = []
       self.opponent_past_utterances = []
       self.repeat_count = 0
       self.utt_count = 0
       if self.twin: self.utt_count = 5
       return "OK"

    # The core of your agent's ability should be implemented here:
    def makeMove(self, currentState, currentRemark, timeLimit=10000):
        #print("makeMove has been called")
        #print("Returning from makeMove")
        who = currentState.whose_move
        self.opponent_past_utterances = []

        startTime = time.time()

        alpha = float('-inf')
        beta = float('inf')
        moveAndscore = self.minimax(currentState, depthRemaining=3, pruning=True, alpha=alpha, beta=beta)

        elapsedTime = time.time() - startTime
        remainingTime = max(0, timeLimit - elapsedTime)

        if remainingTime <= 0:
            print("Time limit exceeded!")

        bestMove = moveAndscore[1]
        if bestMove is None:
            print("No valid move available. This may occur at depth 0 or game end.")
            return [None, "No move possible."]

        newState = generateSuccessor(currentState, bestMove, who)
        print(currentRemark)

        currentState.change_turn()
        if self.twin:
            currentRemark = completionPluto(
            f"Based on the current state {currentState}, respond to {currentRemark} "
                "with a confused and straight-forward comment that reflects your team: O's competitive spirit in one concise sentence")

        else:
           currentRemark = completionGoofy(
            f"Based on the current state {currentState}, respond to {currentRemark} "
                "with a competitive and witty comment that reflects your team X's game's objectives, "
                "strategies, or challenges in one concise sentence.")
        print(currentRemark)

        return [[bestMove, newState], currentRemark]


    # The main adversarial search function:
    def minimax(self,
            state,
            depthRemaining,
            pruning=False,
            alpha=None,
            beta=None,
            zHashing=None):
        # default_score = 0 # Value of the passed-in state. Needs to be computed.

        # return [default_score, "move[i,j]", "more of my stuff"]
        # Only the score is required here but other stuff can be returned
        # in the list, after the score, in case you want to pass info
        # back from recursive calls that might be used in your utterances,
        # etc.
        if state.finished or depthRemaining == 0:
            score = self.staticEval(state)
            return [score, None]

        who = state.whose_move
        bestAction = None

        if who == 'X':
            evalX = float('-inf')
            xActions = legalActions(state,'X')
            xActions.sort(key=lambda action: self.staticEval(generateSuccessor(state, action, state.whose_move)), reverse=True)
            for a in xActions:
            #generate succState(move, who) of each action
                succState = generateSuccessor(state, a, who)
                eval = self.minimax(succState, depthRemaining - 1, pruning, alpha, beta)

                if eval[0] > evalX:
                    evalX = eval[0]
                    bestAction = a
                if pruning:
                    alpha = max(alpha, evalX)
                    if beta <= alpha:
                        break
            return [evalX, bestAction]
        else: #'O' (minimizing player)
            evalO = float('inf')
            oActions = legalActions(state, 'O')
            oActions.sort(key=lambda action: self.staticEval(generateSuccessor(state, action, state.whose_move)), reverse=True)
            for a in oActions:
            #generate succState(move, who) of each action
                succState = generateSuccessor(state, a, who)
                eval = self.minimax(succState, depthRemaining - 1)
                if eval[0] < evalO:
                    evalO = eval[0]
                    bestAction = a
                if pruning:
                    beta = min(beta, evalO)
                    if beta <= alpha:
                        break
            return [evalO, bestAction]

    def staticEval(self, state):
        #print('calling staticEval. Its value needs to be computed!')
        # Values should be higher when the states are better for X,
        # lower when better for O.
        who = state.whose_move
        score = 0
        x_indicies = []
        o_indicies = []
        x_pairs = 0
        o_pairs = 0
        x_count = 0
        o_count = 0

        for row, r in enumerate(state.board):
            for col, c in enumerate(r):
                if c == 'X':
                    #counts of X's or O's
                    x_count += 1
                    x_indicies.append([row, col])
                    #neighbor counts of X's or O's
                    x_pairs += eval_helper(c, state.board, row, col)
                elif c == 'O':
                    o_count += 1
                    o_indicies.append([row, col])
                    o_pairs += eval_helper(c, state.board, row, col)

        score += 5 * (x_pairs - o_pairs)
        score += (x_count - o_count)

        return score

def eval_helper(value, board, row, col):
    pairs = 0
    rows = len(board)
    cols = len(board[0])

    if col + 1 < cols and board[row][col + 1] == value: # right
        pairs += 1
    if col - 1 >= 0 and board[row][col - 1] == value: # left
        pairs += 1
    if row + 1 < rows and board[row + 1][col] == value: # down
        pairs += 1
    if row - 1 >= 0 and board[row - 1][col] == value: # up
        pairs += 1
    if row - 1 >= 0 and col + 1 < cols and board[row - 1][col + 1] == value: # diagonal right up
        pairs += 1
    if row - 1 >= 0 and col - 1 >= 0 and board[row - 1][col - 1] == value: # diagonal left up
        pairs += 1
    if row + 1 < rows and col - 1 >= 0 and board[row + 1][col - 1] == value: # diagonal left down
        pairs += 1
    if row + 1 < rows and col + 1 < cols and board[row + 1][col + 1] == value: # diagonal right down
        pairs += 1

    return pairs


def legalActions(state, who):
    board = state.board
    rows, cols = len(board), len(board[0])
    legal_moves = {(2,0), }

    def is_valid_index(r, c):
        return 0 <= r < rows and 0 <= c < cols

    for r, row in enumerate(board):
        for c, val in enumerate(row):
            if val == ' ':
                legal_moves.add((r, c))

            elif val == who:
                if is_valid_index(r - 1, c) and board[r - 1][c] == ' ':
                    legal_moves.add((r - 1, c))  # North
                if is_valid_index(r + 1, c) and board[r + 1][c] == ' ':
                    legal_moves.add((r + 1, c))  # South
                if is_valid_index(r, c - 1) and board[r][c - 1] == ' ':
                    legal_moves.add((r, c - 1))  # West
                if is_valid_index(r, c + 1) and board[r][c + 1] == ' ':
                    legal_moves.add((r, c + 1))  # East
                if is_valid_index(r - 1, c + 1) and board[r - 1][c + 1] == ' ':
                    legal_moves.add((r - 1, c + 1))  # Diagonal Right-Up
                if is_valid_index(r - 1, c - 1) and board[r - 1][c - 1] == ' ':
                    legal_moves.add((r - 1, c - 1))  # Diagonal Left-Up
                if is_valid_index(r + 1, c - 1) and board[r + 1][c - 1] == ' ':
                    legal_moves.add((r + 1, c - 1))  # Diagonal Left-Down
                if is_valid_index(r + 1, c + 1) and board[r + 1][c + 1] == ' ':
                    legal_moves.add((r + 1, c + 1))  # Diagonal Right-Down

    # print("Current Board State:")
    # print(state)
    # print("Legal Moves:", legal_moves)
    return list(legal_moves)

def generateSuccessor(state, move, who):
    newBoard = deep_copy(state.board)
    #print(f"move: {move}")
    (r, c) = move

    newBoard[r][c] = who

    #create new state obj
    newStateObj = State(old=state)
    newStateObj.board = newBoard

    if not newStateObj.finished:
        newStateObj.change_turn()

    return newStateObj

# OPTIONAL THINGS TO KEEP TRACK OF:

#  WHO_MY_OPPONENT_PLAYS = other(WHO_I_PLAY)
#  MY_PAST_UTTERANCES = []
#  OPPONENT_PAST_UTTERANCES = []
#  UTTERANCE_COUNT = 0
#  REPEAT_COUNT = 0 or a table of these if you are reusing different utterances

#OpenAI API utterances
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key="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")

chat_completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello world"}]
)

def completionGoofy(content, state=''):
  completion = client.chat.completions.create(
      model="gpt-4o",
      messages=[
          {"role": "system", "content": "You are Goofy from Mickey Mouse, a silly and clumsy player with a fun-loving, energetic attitude. You speak like a sports commentator, adding playful, funny remarks as you make your moves in a K-in-a-row game, similar to Tic-Tac-Toe but on a larger board. Keep the mood light, humorous, and entertaining, even when things get competitive. Use lots of enthusiasm and quirky phrases, making the game more fun for everyone! You are team X"},
          {
              "role": "user",
              "content": content + str(state)
          }
      ]
  )
  print(completion.choices[0].message.content)
  return completion.choices[0].message.content

def completionPluto(content, state=''):
  completion = client.chat.completions.create(
      model="gpt-4o",
      messages=[
          {"role": "system", "content": "You are Pluto from Mickey Mouse, a loyal but slightly confused dog who is playing a K-in-a-row game similar to Tic-Tac-Toe but on a larger board. You're straightforward in your actions and responses but tend to get a little puzzled at times. You may not fully understand all the strategies, but you’re determined to do your best! Keep things simple, sincere, and a bit unsure at times, adding humor with your straightforwardness and occasional confusion. You are team O"},
          {
              "role": "user",
              "content": content + str(state)
          }
      ]
  )
  return completion.choices[0].message.content
