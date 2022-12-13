#!/usr/bin/python3

import argparse
from typing import Dict, List, Any
import sys
import time
from stockfish import Stockfish
import numpy
import random
import os, requests, re

import chess
import chess.engine

from stockfish import Stockfish

# this module implement's Tomasz Michniewski's Simplified Evaluation Function
# https://www.chessprogramming.org/Simplified_Evaluation_Function
# note that the board layouts have been flipped and the top left square is A1

sf_depth = 200
sf_time = None

stockfish = Stockfish("C:\\Stockfish\\stockfish.exe")
stockfish.set_elo_rating(3000)
stockfish.set_depth(sf_depth)
stockfish = Stockfish(parameters={'Threads': 4})

OPENING_LENGTH = 8
ENDGAME_COUNT = 12

# fmt: off
piece_value = {
    chess.PAWN: 100,
    chess.ROOK: 500,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.QUEEN: 900,
    chess.KING: 20000
}

pawnEvalWhite = [
    -5, 4, 0, 3, 5, -4,  -5,  -4,
    -2, 6, 7, -16, -15, 5, 2, 5,  
    10, -3, -12, -3, -6, -9, -3, 10, 
    -3, 4, -4, 17, 25, -2, 0, -3,  
    4, 3, 10, 33, 27, 8, 2, 9,  
    13, 13, 23, 33, 29, 17, 13, 13, 
    48, 44, 50, 53, 47, 52, 55, 45, 
    2, -2, -6, -6, -2, -5, -5, -1
]
pawnEvalBlack = list(reversed(pawnEvalWhite))

knightEvalWhite = [
    -54, -45, -28, -29, -31, -30, -37, -54,
    -44, -24, -1, -1, -4, 3, -16, -36, 
    -29, 0, 6, 9, 13, 11, -2, -28,
    -24, 4, 17, 15, 20, 17, 7, -30, 
    -27, -2, 11, 22, 20, 13, -1, -29, 
    -32, 9, 11, 16, 19, 13, 9, -31, 
    -41, -16, -3, 1, 6, 5, -22, -39, 
    -53, -41, -29, -35, -28, -21, -44, -55
]
knightEvalBlack = list(reversed(knightEvalWhite))

bishopEvalWhite = [
    -13, -15, -12, -15, -14, -14, -15, -18, 
    -8, 2, 5, -4, 5, 1, 4, -12, 
    -2, 7, 12, 15, 11, 5, 13, -13, 
    -5, 0, 11, 13, 8, 2, 3, -13, 
    -8, 4, 1, 13, 14, 7, 2, -15, 
    -15, 2, 1, 13, 14, 9, 2, -1, 
    -13, -4, 0, 5, 4, 2, 9, -5, 
    -22, -15, -13, -7, -11, -5, -14, -25
]
bishopEvalBlack = list(reversed(bishopEvalWhite))

rookEvalWhite = [
    3, 5, 3, 6, 4, 4, 6, -3, 
    -6, -1, 1, 1, 2, 0, 4, -1, 
    -1, 4, 5, 2, -3, 4, 0, -4,
    -4, 3, 1, -5, -5, 0, 3, -6, 
    -7, 1, 0, -3, 5, 2, 4, -3, 
    0, 4, 4, -5, -1, -4, -4, -8, 
    2, 7, 5, 11, 10, 7, 15, 2, 
    2, -5, 0, 0, -4, -1, -5, 5
]
rookEvalBlack = list(reversed(rookEvalWhite))

queenEvalWhite = [
    -16, -8, -13, -2, -10, -1, -5, -25, 
    -12, 2, 2, -3, -2, -3, 2, -11, 
    -9, -1, 5, 4, 7, 7, 2, -9, 
    -6, -7, 1, 6, 4, 8, -3, -4, 
    -6, 0, 2, 3, 6, 5, 2, -10, 
    -15, 10, 5, 2, 1, 2, 4, -10, 
    -11, 4, 7, 3, 5, -5, -2, -5, 
    -17, -12, -5, -8, -2, -7, -1, -24
]

queenEvalBlack = list(reversed(queenEvalWhite))

kingEvalWhite = [
    25, 28, 12, 1, -5, 9, 33, 22, 
    20, 10, -4, 0, -9, 1, 25, 25, 
    -11, -17, -20, -24, -18, -17, -16, -15, 
    23, -29, -32, -37, -36, -33, -31, -22, 
    -33, -42, -41, -54, -54, -40, -42, -30, 
    -33, -46, -41, -53, -45, -45, -37, -27, 
    -32, -43, -40, -52, -49, -38, -40, -31, 
    -30, -39, -41, -43, -53, -36, -37, -30
]
kingEvalBlack = list(reversed(kingEvalWhite))

kingEvalEndGameWhite = [
    50, -30, -30, -30, -30, -30, -30, -50,
    -30, -30,  0,  0,  0,  0, -30, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -20, -10,  0,  0, -10, -20, -30,
    -50, -40, -30, -20, -20, -30, -40, -50
]
kingEvalEndGameBlack = list(reversed(kingEvalEndGameWhite))

debug_info: Dict[str, Any] = {}


MATE_SCORE     = 1000000000
MATE_THRESHOLD =  999000000


'''
############################################
############################################
            GAME MAIN SECTION
############################################
############################################
'''

def start():
    """
    Start the command line user interface.
    """

    player_color = input("Enter color: ")
    while player_color not in ['White', 'Black']:
        player_color = input("Please enter valid color (White, Black; case sensitive): ")

    user_side = (
        chess.WHITE if player_color == "White" else chess.BLACK
    )

    in_endgame = False

    ai_color = 'Black' if player_color == 'White' else 'White'

    print(f'When prompted to move, enter commands like "a2a4"')
    board = chess.Board()
    movelist = []
    
    # 8 moves using database online
    board, movelist = play_opening(board, movelist, player_color)

    # since opening function always returns to white player, check if plyer is white
    move = None
    if user_side == chess.WHITE and board.turn:
        print(board)
        move = get_player_move(board, movelist)
        board.push(move)

    # ai moves then player moves
    while not board.is_game_over():

        if not in_endgame:

            # PST based MinMax
            move = next_move(get_depth(), board, debug=False)

        # if in endgame
        else:
            # stockfish
            move = get_ai_move(board)

        # push ai move
        board.push(chess.Move.from_uci(move) if in_endgame else move)
        movelist.append(move)
        if board.is_game_over():
            print(get_result(board, ai_color))
            return movelist

        # player moves
        move = get_player_move(board, movelist)
        board.push(chess.Move.from_uci(move))
        movelist.append(move)
        if board.is_game_over():
            print(get_result(board, player_color))
            return movelist

        # endgame defined as 12 pieces or less
        # checking when ai makes the move is irrelevant as the move is made while on the old logic
        if (num_pieces_from_fen(board.board_fen()) <= ENDGAME_COUNT):
            in_endgame = True
            print('\n\nSwitched to Endgame\n\n')

    print(f"\nResult: [w] {board.result()} [b]")

    return movelist

'''
############################################
############################################
            MAIN SECTION END
############################################
############################################
'''

def get_result(board, color):
    if board.is_checkmate():
        return f'{color} won by checkmate:\n{board}'

    if board.is_stalemate():
        return f'{color} drew by stalemate:\n{board}'

    if board.is_insufficient_material():
        return f'{color} drew by insufficient material:\n{board}'

    return f'error in determining outcome:\n{board}'

def get_player_move(board, movelist):
    move = input(f'\n{board}\nAI\'s move: {movelist[-1] if movelist else None}\nEnter move: ')
    match = re.match('([a-h][1-8])'*2, move)
    while not match or chess.Move.from_uci(move) not in board.legal_moves:
        move = input(f'Please enter a legal move: ')
        match = re.match('([a-h][1-8])'*2, move)
    return move

def get_depth() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", default=3, help="provide an integer (default: 3)")
    args = parser.parse_args()
    return max([1, int(args.depth)])

def fish_eval(fen):
    stockfish.set_fen_position(fen)
    return int(stockfish.get_evaluation()['value'])

'''
############################################
############################################
                OPENING SECTION
############################################
############################################
'''

def play_opening(board, movelist, player_color):

    ai_has_moved = False

    while not movelist or len(movelist) < OPENING_LENGTH:

        # go first if white or skip player turn and let ai go
        if (board.turn and player_color.lower() == 'white') or ai_has_moved:

            move = get_player_move(board, movelist)
            movelist.append(move)
            board.push(chess.Move.from_uci(move))

            if board.is_game_over():
                print(get_result(board, player_color))
                return

        moves, movelist = getMovesFromOpening(movelist, not (len(movelist)%2), board)

        # if no moves found in database switch to regular play
        if not moves:
            print("End of opening - no moves in database")
            return board, movelist
        choice = random.randint(0, len(moves)-1)
        movelist.append(moves[choice])
        board.push(chess.Move.from_uci(moves[choice]))
        ai_has_moved = True

    # if no moves found in database switch to regular play
    
    print(f'End of opening - move limit {OPENING_LENGTH} reached')

    return board, movelist

# 0 = black, 1 = white
def getMovesFromOpening(movelist, white, board):
    #obtain the content of the URL in HTML
    #url = "https://www.gopoly.com/sports/mbkb/2018-19/schedule"
    url = f'https://explorer.lichess.ovh/lichess?speeds=rapid,classical&ratings=2500&play={",".join(movelist)}'
    response = requests.get(url)
    data = response.json()

    # parse through opening explorer
    moves = []
    for item in data['moves']:

        # calculate win ratio for white
        ratio = int(item['white'])/int(item['black']) if item['black'] else float('inf')
        moves.append((item['uci'], ratio))

    # sort moves by win ratio, ascending if black descending if white
    moves.sort(key = lambda _: _[1], reverse=white)

    # return top 3 moves that have the best win ratio for the color of the player
    return [move[0] for move in moves[0:3]], movelist

'''
############################################
############################################
        MIDDLEGAME EVALUATE SECTION
############################################
############################################
'''

def move_value(board: chess.Board, move: chess.Move, endgame: bool) -> float:
    """
    How good is a move?
    A promotion is great.
    A weaker piece taking a stronger piece is good.
    A stronger piece taking a weaker piece is ---bad--- not as good.
    Also consider the position change via piece-square table.
    """
    if move.promotion is not None:
        return -float("inf") if board.turn == chess.BLACK else float("inf")

    _piece = board.piece_at(move.from_square)
    if _piece:
        _from_value = evaluate_piece(_piece, move.from_square, endgame)
        _to_value = evaluate_piece(_piece, move.to_square, endgame)
        position_change = _to_value - _from_value
    else:
        raise Exception(f"A piece was expected at {move.from_square}")

    capture_value = 0.0
    if board.is_capture(move):
        capture_value = evaluate_capture(board, move)

    current_move_value = capture_value + position_change
    if board.turn == chess.BLACK:
        current_move_value = -current_move_value

    return current_move_value


def evaluate_capture(board: chess.Board, move: chess.Move) -> float:
    """
    Given a capturing move, weight the trade being made.
    """
    if board.is_en_passant(move):
        return piece_value[chess.PAWN]
    _to = board.piece_at(move.to_square)
    _from = board.piece_at(move.from_square)
    if _to is None or _from is None:
        raise Exception(
            f"Pieces were expected at _both_ {move.to_square} and {move.from_square}"
        )

    # code below changed to encourage captures, not discourage captures if going from high->low
    captured = piece_value[_to.piece_type]
    capturing = piece_value[_from.piece_type]
    captured_more_valueable = captured > capturing

    # if the captured is worth more than capturing, return difference (9, 4 -> 5)
    # if the captured is worth less than capturing, return division [(4, 9 -> 4/9) is better than (1, 9 -> 1/9)]
    return (captured - capturing) if captured_more_valueable else (captured/capturing)


def evaluate_piece(piece: chess.Piece, square: chess.Square, end_game: bool) -> int:
    piece_type = piece.piece_type
    mapping = []
    if piece_type == chess.PAWN:
        mapping = pawnEvalWhite if piece.color == chess.WHITE else pawnEvalBlack
    if piece_type == chess.KNIGHT:
        mapping = knightEvalWhite if piece.color == chess.WHITE else knightEvalBlack
    if piece_type == chess.BISHOP:
        mapping = bishopEvalWhite if piece.color == chess.WHITE else bishopEvalBlack
    if piece_type == chess.ROOK:
        mapping = rookEvalWhite if piece.color == chess.WHITE else rookEvalBlack
    if piece_type == chess.QUEEN:
        mapping = queenEvalWhite if piece.color == chess.WHITE else queenEvalBlack
    if piece_type == chess.KING:
        # use end game piece-square tables if neither side has a queen
        if end_game:
            mapping = (
                kingEvalEndGameWhite
                if piece.color == chess.WHITE
                else kingEvalEndGameBlack
            )
        else:
            mapping = kingEvalWhite if piece.color == chess.WHITE else kingEvalBlack

    return mapping[square]


def evaluate_board(board: chess.Board) -> float:
    """
    Evaluates the full board and determines which player is in a most favorable position.
    The sign indicates the side:
        (+) for white
        (-) for black
    The magnitude, how big of an advantage that player has
    """
    total = 0
    end_game = check_end_game(board)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if not piece:
            continue

        value = piece_value[piece.piece_type] + evaluate_piece(piece, square, end_game)
        total += value if piece.color == chess.WHITE else -value

    return total


def check_end_game(board: chess.Board) -> bool:
    """
    Are we in the end game?
    Per Michniewski:
    - Both sides have no queens or
    - Every side which has a queen has additionally no other pieces or one minorpiece maximum.
    """
    queens = 0
    minors = 0

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.piece_type == chess.QUEEN:
            queens += 1
        if piece and (
            piece.piece_type == chess.BISHOP or piece.piece_type == chess.KNIGHT
        ):
            minors += 1

    if queens == 0 or (queens == 2 and minors <= 1):
        return True

    return False

'''
############################################
############################################
        MIDDLEGAME MOVEMENT SECTION
############################################
############################################
'''

def next_move(depth: int, board: chess.Board, debug=True) -> chess.Move:
    """
    What is the next best move?
    """
    debug_info.clear()
    debug_info["nodes"] = 0
    t0 = time.time()

    move = minimax_root(depth, board)

    debug_info["time"] = time.time() - t0
    if debug == True:
        print(f"info {debug_info}")
    return move


def get_ordered_moves(board: chess.Board) -> List[chess.Move]:
    """
    Get legal moves.
    Attempt to sort moves by best to worst.
    Use piece values (and positional gains/losses) to weight captures.
    """
    end_game = check_end_game(board)

    def orderer(move):
        return move_value(board, move, end_game)

    in_order = sorted(
        board.legal_moves, key=orderer, reverse=(board.turn == chess.WHITE)
    )
    return list(in_order)


def minimax_root(depth: int, board: chess.Board) -> chess.Move:
    """
    What is the highest value move per our evaluation function?
    """
    # White always wants to maximize (and black to minimize)
    # the board score according to evaluate_board()
    maximize = board.turn == chess.WHITE
    best_move = -float("inf")
    if not maximize:
        best_move = float("inf")

    moves = get_ordered_moves(board)
    best_move_found = moves[0]

    for move in moves:
        board.push(move)
        # Checking if draw can be claimed at this level, because the threefold repetition check
        # can be expensive. This should help the bot avoid a draw if it's not favorable
        # https://python-chess.readthedocs.io/en/latest/core.html#chess.Board.can_claim_draw
        if board.can_claim_draw():
            value = 0.0
        else:
            value = minimax(depth - 1, board, -float("inf"), float("inf"), not maximize)
        board.pop()
        if maximize and value >= best_move:
            best_move = value
            best_move_found = move
        elif not maximize and value <= best_move:
            best_move = value
            best_move_found = move

    return best_move_found


def minimax(
    depth: int,
    board: chess.Board,
    alpha: float,
    beta: float,
    is_maximising_player: bool,
) -> float:
    """
    Core minimax logic.
    https://en.wikipedia.org/wiki/Minimax
    """
    debug_info["nodes"] += 1

    if board.is_checkmate():
        # The previous move resulted in checkmate
        return -MATE_SCORE if is_maximising_player else MATE_SCORE
    # When the game is over and it's not a checkmate it's a draw
    # In this case, don't evaluate. Just return a neutral result: zero
    elif board.is_game_over():
        return 0

    if depth == 0:
        return evaluate_board(board)

    if is_maximising_player:
        best_move = -float("inf")
        moves = get_ordered_moves(board)
        for move in moves:
            board.push(move)
            curr_move = minimax(depth - 1, board, alpha, beta, not is_maximising_player)
            # Each ply after a checkmate is slower, so they get ranked slightly less
            # We want the fastest mate!
            if curr_move > MATE_THRESHOLD:
                curr_move -= 1
            elif curr_move < -MATE_THRESHOLD:
                curr_move += 1
            best_move = max(
                best_move,
                curr_move,
            )
            board.pop()
            alpha = max(alpha, best_move)
            if beta <= alpha:
                return best_move
        return best_move
    else:
        best_move = float("inf")
        moves = get_ordered_moves(board)
        for move in moves:
            board.push(move)
            curr_move = minimax(depth - 1, board, alpha, beta, not is_maximising_player)
            if curr_move > MATE_THRESHOLD:
                curr_move -= 1
            elif curr_move < -MATE_THRESHOLD:
                curr_move += 1
            best_move = min(
                best_move,
                curr_move,
            )
            board.pop()
            beta = min(beta, best_move)
            if beta <= alpha:
                return best_move
        return best_move

'''
############################################
############################################
            ENDGAME SECTION
############################################
############################################
'''

def num_pieces_from_fen(fen):

  num = 0

  for char in fen:
    if char.isdigit() or char == '/':
      pass
    else:
      num += 1

  return num

def get_result(board, color):
    if board.is_checkmate():
        return f'{color} won by checkmate:\n{board}'

    if board.is_stalemate():
        return f'{color} drew by stalemate:\n{board}'

    if board.is_insufficient_material():
        return f'{color} drew by insufficient material:\n{board}'

    return f'error in determining outcome:\n{board}'


def get_ai_move(board):
    stockfish.set_fen_position(board.fen())

    print("\nThinking...\n")

    if sf_time != None:
        top_move = stockfish.get_best_move_time(sf_time * 1000)
    else:
        top_move = stockfish.get_best_move()
    #Get the best move
    # Move the chess piece based on the best move
    return top_move

'''
############################################
############################################
            IF MAIN SECTION
############################################
############################################
'''

if __name__ == "__main__":
    try:
        start()
    except KeyboardInterrupt:
        pass
