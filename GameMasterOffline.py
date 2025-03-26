'''GameMasterOffline.py based on GameMaster.py

 Updated Nov. 17, 2024.
 See the test function at the end for how to customize
 the runs: choice of games and agents.

(C) University of Washington and S. Tanimoto, 2024.

'''

from time import sleep
USE_HTML = True
if USE_HTML: import gameToHTML

from winTesterForK import winTesterForK

from game_types import TTT, FIAR, Cassini

TIME_PER_MOVE = 1.0 # In seconds
INITIAL_STATE = TTT.initial_state


# Establish global variables, with defaults for now.
K = None
N = None
M = None
TURN_LIMIT = None

# To be called from WebGameAgent if using the web:
def set_game(game_type):
    global K, GAME_TYPE, TURN_LIMIT, N, M, INITIAL_STATE
    K = game_type.k
    N = game_type.n
    M = game_type.m
    GAME_TYPE = game_type
    TURN_LIMIT = game_type.turn_limit
    INITIAL_STATE = game_type.initial_state

PLAYERX = None
PLAYERO = None
def set_players(px, po):
    global PLAYERX, PLAYERO
    PLAYERX = px
    PLAYERO = po

FINISHED = False
def runGame():
    currentState = INITIAL_STATE
    player1 = PLAYERX
    player2 = PLAYERO
    renderCommentary('The Gamemaster says, "Players, introduce yourselves."')
    renderCommentary('     (Playing X:) '+player1.introduce())
    renderCommentary('     (Playing O:) '+player2.introduce())

    if USE_HTML:
        gameToHTML.startHTML(player1.nickname, player2.nickname, GAME_TYPE.short_name, 1)
    try:
        p1comment = player1.prepare(GAME_TYPE, 'X', player2.nickname)
    except Exception as e:
        print("Failed to prepare perhaps because: ", e)
        report = 'Player 1 ('+player1.nickname+' failed to prepare, and loses by default.'
        renderCommentary(report)
        if USE_HTML: gameToHTML.reportResult(report)
        report = 'Congratulations to Player 2 ('+player2.nickname+')!'
        renderCommentary(report)
        if USE_HTML: gameToHTML.reportResult(report)
        if USE_HTML: gameToHTML.endHTML()
        return
    try:
        p2comment = player2.prepare(GAME_TYPE, 'O', player1.nickname)
    except Exception as e:
        print("Failed to prepare perhaps because: ", e)
        report = 'Player 2 ('+player2.nickname+' failed to prepare, and loses by default.'
        renderCommentary(report)
        if USE_HTML: gameToHTML.reportResult(report)
        report = 'Congratulations to Player 1 ('+player1.nickname+')!'
        renderCommentary(report)
        if USE_HTML: gameToHTML.reportResult(report)
        if USE_HTML: gameToHTML.endHTML()
        return
        return

    renderCommentary('The Gamemaster says: We\'re playing '+GAME_TYPE.long_name+'.')
    renderCommentary('The Gamemaster says: Let\'s Play!')
    renderCommentary('The initial state is...')

    currentRemark = "The game is starting."
    if USE_HTML: gameToHTML.stateToHTML(currentState)
    XsTurn = True
    name = None
    global FINISHED
    FINISHED = False
    turnCount = 0
    printState(currentState)
    while not FINISHED:
        who = currentState.whose_move
        if XsTurn:
            playerResult = player1.makeMove(currentState, currentRemark, TIME_PER_MOVE)
            name = player1.nickname
            XsTurn = False
        else:
            playerResult = player2.makeMove(currentState, currentRemark, TIME_PER_MOVE)
            name = player2.nickname
            XsTurn = True
        moveAndState, currentRemark = playerResult
        if moveAndState==None:
            FINISHED = True; continue
        move, currentState = moveAndState
        moveReport = "Move is by "+who+" to "+str(move)
        renderCommentary(moveReport)
        utteranceReport = name +' says: '+currentRemark
        renderCommentary(utteranceReport)
        if USE_HTML: gameToHTML.reportResult(moveReport)
        if USE_HTML: gameToHTML.reportResult(utteranceReport)
        possibleWin = winTesterForK(currentState, move, K)
        if possibleWin != "No win":
            FINISHED = True
            currentState.finished = True
            printState(currentState)
            if USE_HTML: gameToHTML.stateToHTML(currentState, finished=True)
            renderCommentary(possibleWin)
            if USE_HTML: gameToHTML.reportResult(possibleWin)
            if USE_HTML: gameToHTML.endHTML()
            return
        printState(currentState)
        if USE_HTML: gameToHTML.stateToHTML(currentState)
        turnCount += 1
        if turnCount == TURN_LIMIT: FINISHED=True
        else:
            sleep(WAIT_TIME_AFTER_MOVES) # NOT TOO FAST.
    printState(currentState)
    if USE_HTML: gameToHTML.stateToHTML(currentState)
    who = currentState.whose_move
    renderCommentary("Game over; it's a draw.")
    if USE_HTML: gameToHTML.reportResult("Game Over; it's a draw")
    if USE_HTML: gameToHTML.endHTML()

def printState(s):
    global FINISHED
    board = s.board
    who = s.whose_move
    horizontalBorder = "+"+3*M*"-"+"+"
    renderCommentary(horizontalBorder)
    for row in board:
        line = "|"
        for item in row:
            line += " "+item+" "
        line += "|"
        renderCommentary(line)
    renderCommentary(horizontalBorder)
    if not FINISHED:
      renderCommentary("It is "+who+"'s turn to move.\n")

# Temporary function.  Remove when other channels are working.
def renderCommentary(stuff):
    print(stuff)

def render_move_and_state(move, state):
   # NOTE: THIS DEFN WILL BE OVERWRITTEN WHEN USED ON THE WEB.
   print(move, state)

def render_utterance(who, utterance):
   # NOTE: THIS DEFN WILL BE OVERWRITTEN WHEN USED ON THE WEB.
   print(who+' says: '+utterance)

# Not used in offline version:
#def async_runGame():
#    fut = ensure_future(runGame())

WAIT_TIME_AFTER_MOVES = 0.01
def set_wait_time(t):
    global WAIT_TIME_AFTER_MOVES
    WAIT_TIME_AFTER_MOVES = float(t)

def test():
    # Stand-alone test
    print("Starting stand-alone test of GameMaster.py")
    # Edit this to change what version of K-in-a-Row is used.
    set_game(FIAR) # default is Tic-Tac-Toe
    #set_game(FIAR) # Five in a Row
    # Import 1 or 2 agent files here.
    # If using only 1, create 2 instances of it, one of
    # which is a "twin".

    #import yourUWNetID_KInARow as h
    import raghavip_KInARow as h
    import achen47_katzhang_KInARow as hh
    px = h.OurAgent()
    po = hh.OurAgent()
    set_players(px, po)
    print("Players are set.")
    print("Now let's run the game.")
    runGame()

if __name__ == '__main__':
    test()

