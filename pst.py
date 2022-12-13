# inspired by the https://github.com/thomasahle/sunfish user inferface

import chess
import argparse
from typing import Dict, List, Any
import chess
import sys
import time
from original_pst import *
from stockfish import Stockfish
import pygad
import numpy
import random

# this module implement's Tomasz Michniewski's Simplified Evaluation Function
# https://www.chessprogramming.org/Simplified_Evaluation_Function
# note that the board layouts have been flipped and the top left square is A1

sf_depth = 200
sf_time = None

stockfish = Stockfish("C:\\Stockfish\\stockfish.exe")
stockfish.set_elo_rating(3000)
stockfish.set_depth(sf_depth)
stockfish = Stockfish(parameters={'Threads': 4})

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
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10, -20, -20, 10, 10,  5,
    5, -5, -10,  0,  0, -10, -5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5,  5, 10, 25, 25, 10,  5,  5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
    0, 0, 0, 0, 0, 0, 0, 0
]
pawnEvalBlack = list(reversed(pawnEvalWhite))

knightEvalWhite = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50
]
knightEvalBlack = list(reversed(knightEvalWhite))

bishopEvalWhite = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20
]
bishopEvalBlack = list(reversed(bishopEvalWhite))

rookEvalWhite = [
    0, 0, 0, 5, 5, 0, 0, 0,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    5, 10, 10, 10, 10, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0
]
rookEvalBlack = list(reversed(rookEvalWhite))

queenEvalWhite = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5,
    0, 0, 5, 5, 5, 5, 0, -5,
    -10, 5, 5, 5, 5, 5, 0, -10,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20
]

queenEvalBlack = list(reversed(queenEvalWhite))

kingEvalWhite = [
    20, 30, 10, 0, 0, 10, 30, 20,
    20, 20, 0, 0, 0, 0, 20, 20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30
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
            GENETIC ALGO SECTION 1
############################################
############################################
'''

GLOBAL_PST = [ori_pst[key] for key in ori_pst]

original = numpy.asarray(GLOBAL_PST).flatten()
init_pop = [[(item + random.randint(-5, 5)) for item in original] for i in range(10)]

def fitness_func(solution, solution_idx):
    solution_to_pst(solution)

    total_score = 0
   
    board = chess.Board()

    number_of_moves = 0
    while not board.is_game_over() and number_of_moves < 80:        
        board.push(next_move(get_depth(), board, debug=False))

        # # print every 10th move
        # if not (number_of_moves % 10):
        #     print(number_of_moves)
        #     print(board)

        number_of_moves += 1

    print(board)
    print(f"\nResult in {number_of_moves} moves: [w] {board.result()} [b]")

    # we want checkmates and good positions to be very highly valued
    evaluation = fish_eval(board.fen())
    if board.result() == '1/2-1/2' or not board.is_game_over():

        if evaluation < 0:

            # -500 -> -inf
            if evaluation <= -500:
                total_score -= 9
            # -201 -> -500
            elif evaluation >= -500 and -200 > evaluation:
                total_score -= 7

            # 0 -> -200
            elif evaluation >= -200:
                total_score -= 4

        # white advantage
        else:
            # 500 -> inf
            if evaluation >= 500:
                total_score += 9
            # 201 -> 500
            elif evaluation <= 500 and 200 < evaluation:
                total_score += 7

            # 0 -> 200
            elif evaluation <= 200:
                total_score += 4

    elif board.result() == '0-1':
        total_score -= 10

    # severely punish being mated
    elif board.result() == '1-0':
        total_score += 40

    print(f'Score from eval (lower is better) = {total_score}')
    # number of moves is not that important but the longer they take to checkmate = worse
    total_score += number_of_moves / 2

    if total_score < 5:
        total_score = 5
    # perfect score being 5-10 -> -10 (checkmate) 
    #                               in ~30 moves (1 each player) each round (15)
    # obviously not a perfect classifier but it is very difficult to classify chess!  
    divisor = abs(total_score - 5)
    if divisor < 1:
        divisor = 1.00001

    fitness_score = 1/divisor

    global ga_instance
    print(f'Generation {ga_instance.generations_completed}')
    print(f'Child {solution_idx} stockfish evaluation in {number_of_moves} moves: {evaluation}')
    print(f'Child {solution_idx} total points = {total_score}, divisor = {divisor}. Achieved a score of {fitness_score} using 1/{divisor}')
    return fitness_score

ga_instance = pygad.GA(
                       initial_population=init_pop,
                       num_generations = 7,
                       fitness_func=fitness_func,
                       sol_per_pop=10,
                       num_parents_mating=3,
                       num_genes=384,
                       keep_parents = 3,
                       crossover_type = 'uniform',
                       mutation_type = 'random',
                       mutation_percent_genes = 10,
                       gene_type=numpy.int32,
                       random_mutation_min_val = -5,
                       random_mutation_max_val = 5)

'''
############################################
############################################
            ORIGINAL MAIN SECTION
############################################
############################################
'''

def start():
    """
    Start the command line user interface.
    """
    
    total_score = 0
    for i in range(3):
        board = chess.Board()

        number_of_moves = 0
        while not board.is_game_over() or number_of_moves < 100:        
            board.push(next_move(get_depth(), board, debug=False))
            print(board)
            number_of_moves += 1

        print(f"\nResult: [w] {board.result()} [b]")
        if board.winner == 1:
            total_score += 1

        if board.winner == None:
            evaluation = fish_eval(board.fen())
            if evaluation > 0:
                total_score += 0.7

            else:
                total_score -= 0.2

    if total_score < 0:
        total_score = 0.0001

    return 1/(3 - total_score)


def get_move(board: chess.Board) -> chess.Move:
    """
    Try (and keep trying) to get a legal next move from the user.
    Play the move by mutating the game board.
    """
    move = input(f"\nYour move (e.g. {list(board.legal_moves)[0]}):\n")

    for legal_move in board.legal_moves:
        if move == str(legal_move):
            return legal_move
    return get_move(board)


def get_depth() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", default=3, help="provide an integer (default: 3)")
    args = parser.parse_args()
    return max([1, int(args.depth)])

'''
############################################
############################################
            GENETIC ALGO SECTION 2
############################################
############################################
'''

def start_GAD():

    global ga_instance

    ga_instance.run()
    ga_instance.save(filename='genetic')
    best_solution, solution_fitness, solution_idx = ga_instance.best_solution()
    with open("best_solution.txt", 'w+') as f:

        arr = numpy.asarray(best_solution).reshape(6, 64)
        out = f'{arr}'
        print(arr)
        f.write(out)

    with open("sol_fit.txt", "w+") as f2:
        out = f'{solution_fitness}'
        print(out)
        f2.write(out)


def solution_to_pst(solution):
    # P N B R Q K
    solution_pst = numpy.asarray(solution).reshape(6, 64)
    
    global pawnEvalBlack
    global knightEvalBlack
    global bishopEvalBlack
    global rookEvalBlack
    global queenEvalBlack
    global knightEvalBlack

    pawnEvalBlack = list(reversed(solution_pst[0]))
    knightEvalBlack = list(reversed(solution_pst[1]))
    bishopEvalBlack = list(reversed(solution_pst[2]))
    rookEvalBlack = list(reversed(solution_pst[3]))
    queenEvalBlack = list(reversed(solution_pst[4]))
    kingEvalBlack = list(reversed(solution_pst[5]))

    print("Converted black PST")

def fish_eval(fen):
    stockfish.set_fen_position(fen)
    return int(stockfish.get_evaluation()['value'])

'''
############################################
############################################
            EVALUATE SECTION
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
            MOVEMENT SECTION
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
            ORIGINAL PST SECTION
############################################
############################################
'''


ori_pst = {
    'P': [ 0,  0,  0,  0,  0,  0,  0,  0,
            5, 10, 10, -20, -20, 10, 10,  5,
            5, -5, -10,  0,  0, -10, -5,  5,
            0,  0,  0, 20, 20,  0,  0,  0,
            5,  5, 10, 25, 25, 10,  5,  5,
            10, 10, 20, 30, 30, 20, 10, 10,
            50, 50, 50, 50, 50, 50, 50, 50,
            0, 0, 0, 0, 0, 0, 0, 0
            ],
    'N': [ -50, -40, -30, -30, -30, -30, -40, -50,
            -40, -20, 0, 0, 0, 0, -20, -40,
            -30, 0, 10, 15, 15, 10, 0, -30,
            -30, 5, 15, 20, 20, 15, 5, -30,
            -30, 0, 15, 20, 20, 15, 0, -30,
            -30, 5, 10, 15, 15, 10, 5, -30,
            -40, -20, 0, 5, 5, 0, -20, -40,
            -50, -40, -30, -30, -30, -30, -40, -50
    ],
    'B': [ -20, -10, -10, -10, -10, -10, -10, -20,
            -10, 5, 0, 0, 0, 0, 5, -10,
            -10, 10, 10, 10, 10, 10, 10, -10,
            -10, 0, 10, 10, 10, 10, 0, -10,
            -10, 5, 5, 10, 10, 5, 5, -10,
            -10, 0, 5, 10, 10, 5, 0, -10,
            -10, 0, 0, 0, 0, 0, 0, -10,
            -20, -10, -10, -10, -10, -10, -10, -20
        ],
    'R': [  0, 0, 0, 5, 5, 0, 0, 0,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            5, 10, 10, 10, 10, 10, 10, 5,
            0, 0, 0, 0, 0, 0, 0, 0
        ],
    'Q': [ 
        -20, -10, -10, -5, -5, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 5, 5, 5, 0, -10,
        -5, 0, 5, 5, 5, 5, 0, -5,
        0, 0, 5, 5, 5, 5, 0, -5,
        -10, 5, 5, 5, 5, 5, 0, -10,
        -10, 0, 5, 0, 0, 0, 0, -10,
        -20, -10, -10, -5, -5, -10, -10, -20
        ],
    'K': [   20, 30, 10, 0, 0, 10, 30, 20,
            20, 20, 0, 0, 0, 0, 20, 20,
            -10, -20, -20, -20, -20, -20, -20, -10,
            20, -30, -30, -40, -40, -30, -30, -20,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30
        ],
}

if __name__ == "__main__":
    try:
        start_GAD()
    except KeyboardInterrupt:
        pass
