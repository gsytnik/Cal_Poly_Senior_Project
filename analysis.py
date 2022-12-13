#!/usr/bin/python3

import chess
import chess.engine
import random
from stockfish import Stockfish
import sys

# find number of pieces when using only minmax makes sense over using a neural net
#  number of pieces left and finishing game

sf_depth = 200
sf_time = None

stockfish = Stockfish("C:\\Stockfish\\stockfish.exe")
stockfish.set_elo_rating(3000)
stockfish.set_depth(sf_depth)
stockfish = Stockfish(parameters={'Threads': 4})

num_pieces = 9

csv_files = {
  8: 'eight_piece.csv',
  9: 'nine_piece.csv',
  10: 'ten_piece.csv',
  11: 'eleven_piece.csv',
  12: 'twelve_piece.csv'
}

'''
CSV FILE FORMAT:

Starting fen, Reduction found, Ending fen, Number of Moves, Checkmate?, Stockfish Depth, Stockfish Time Constraint
'''

# positions to evaluate
num_positions = 10

squares_index = {
  'a': 0,
  'b': 1,
  'c': 2,
  'd': 3,
  'e': 4,
  'f': 5,
  'g': 6,
  'h': 7
}

balance_dict = {
  'q': [['q'], ['q'], ['q'], ['r', 'r'], ['n', 'n', 'b'], ['b', 'b', 'n'], ['r', 'n', 'p'], ['r', 'b', 'p']],
  'r': [['p', 'n'], ['p', 'b'], ['r']],
  'n': [['b'], ['n'], ['p', 'p', 'p']],
  'b': [['b'], ['n'], ['p', 'p', 'p']],
  'p': [['p']]
}

rolls = ['p', 'b', 'n', 'r', 'q']

unweighted_rolls = ['p', 'b', 'b', 'b', 'n', 'n', 'n', 'r', 'r', 'q']

def main(args):

  # pos = random_pos_weighted(num_pieces)
  # fen = fen_from_board(pos)

  # board = chess.Board(fen)

  # print(f'Weighted:\n\n{board}')

  if len(args) != 4:
    print(f'Usage: py analysis.py <num_positions> <num_pieces> <depth> <time in seconds>')
    return

  for i in range(3):
    try:
      args[i] = int(args[i])
    except ValueError:
      print(f'Arg {args[i]} must be of type int')
      return

  num_positions = args[0]
  num_pieces = args[1]
  sf_depth = args[2]
  sf_time = args[3]

  if num_pieces not in csv_files.keys():
    print(f'no csv_file entry for {num_pieces} pieces')
    return

  if args[3] == 'None':
    sf_time = None
  else:
    try:
      int(sf_time)
    except ValueError:
      print(f'time in seconds must be None or int.')
      return


  f = open(csv_files[num_pieces], 'a')


  for i in range(num_positions):

    board, fen = gen_board(num_pieces)
    print(f'\nUnweighted:\n\n{board}\nNumber of pieces: ' + \
      f'{num_pieces_from_fen(board.board_fen())} \n{"white" if board.turn else "black"} to move' + \
      f'\nStarting eval: {evaluate(board.fen())}') 
  
    f.write(play_board(board, fen, i + 1))
      


  f.close()

  return

def gen_board(num_pieces):
  pos = balanced_pos(num_pieces)
  fen = fen_from_board(pos)  
  board = chess.Board(fen)
  board, legal = legal_pos(board)
  fen = board.fen()

  while board.is_game_over() or not board.is_valid() or evaluate(fen) > 300 or evaluate(fen) < -300:
    pos = balanced_pos(num_pieces)
    fen = fen_from_board(pos)  
    board = chess.Board(fen)
    board, legal = legal_pos(board)
    fen = board.fen()

  return board, fen


def evaluate(fen):
  stockfish.set_fen_position(fen)
  return int(stockfish.get_evaluation()['value'])


# checks if starting pos of board is legal
def legal_pos(board):
  opposite = not board.turn

  if not board.is_valid():
    turn_fen =' b - - 1 0' if board.turn else ' w - - 0 1'
    board = chess.Board(board.board_fen() + turn_fen)

    if not board.is_valid():
      return board, False

  return board, True


def play_board(board, fen, game):
  num_moves = 0

  # 100 move limit
  while num_moves < 100:

    # get top stockfish move
    num_moves += 1
    get_ai_move(board)
    print(f'{board}\nGame: {game}\nMove: {num_moves}/100\n')

    # if there are 7 or less pieces write to file true and break
    if num_pieces_from_fen(board.board_fen()) <= 7 or board.is_game_over():
      return (f'{fen}, {True}, {board.board_fen()}, {num_moves}, {board.is_game_over()}, {sf_depth}, {sf_time}\n')


  return (f'{fen}, {False}, {board.board_fen()}, {num_moves}, {board.is_game_over()}, {sf_depth}, {sf_time}\n')


def get_ai_move(board):
  stockfish.set_fen_position(board.fen())

  print("\nThinking...\n")

  if sf_time != None:
    top_move = stockfish.get_best_move_time(sf_time * 1000)
  else:
    top_move = stockfish.get_best_move()
  #Get the best move
  # Move the chess piece based on the best move
  board.push(chess.Move.from_uci(top_move))


def num_pieces_from_fen(fen):

  num = 0

  for char in fen:
    if char.isdigit() or char == '/':
      pass
    else:
      num += 1

  return num


def fen_from_board(board):
  fenlist = []

  # for each row
  for row in board:

    # start a new fen str segment
    cur_str = ''
    num_spaces = 0

    # for each item in row
    for i, item in enumerate(row):

      # if it is a 0, that means that it is empty
      if item == 0:

        # add space
        num_spaces += 1

      # otherwise
      else:

        # add numspaces to string if numspaces is more than zero
        if num_spaces > 0:
          cur_str += f'{num_spaces}'

          # set num_spaces to zero again
          num_spaces = 0

        # add piece to fen
        cur_str += item

    if num_spaces > 0:
      cur_str += f'{num_spaces}'

    # add fen str of row
    fenlist.append(cur_str)

  b_fen = '/'.join(fenlist) 

  move = random.randint(0, 1)

  turn_fen =' w - - 1 0' if move == 0 else ' b - - 0 1'

  return b_fen + turn_fen


def random_pos_weighted(num_pieces):
  board = [[0] * 8 for i in range(8)]

  num_rolls = (num_pieces//2) -1
  maxm = len(rolls) - 1
  minm = 0

  # occupied board spaces
  occupied = []

  assign_kings(occupied, board)

  for i in range(num_rolls):
    num = roll_range(minm, maxm)
    if num == 0:
      minm = 1
    if num == len(rolls) - 1:
      maxm -= 1

    piece = rolls[num]
    piece2 = piece.upper()

    pos, row, col = rand_board_idx(occupied)
    occupied.append(pos)
    board[row][col] = piece

    pos, row, col = rand_board_idx(occupied)
    occupied.append(pos)
    board[row][col] = piece2

  return board


def random_pos_unweighted(num_pieces):
  board = [[0] * 8 for i in range(8)]

  num_rolls = num_pieces - 2
  maxm = len(unweighted_rolls) - 1
  minm = 0

  # occupied board spaces
  occupied = []

  assign_kings(occupied, board)

  start = random.randint(0, 1)

  side = 1 if start == 0 else -1

  for i in range(num_rolls):

    num = roll_range(minm, maxm)
    
    piece = unweighted_rolls[num].upper() if side == 1 else unweighted_rolls[num]

    side *= -1


    pos, row, col = rand_board_idx(occupied)

    while ((piece == 'p' or piece == 'P') and (row == 0 or row == 7)):
      pos, row, col = rand_board_idx(occupied)

    occupied.append(pos)

    board[row][col] = piece

  return board


def balanced_pos(num_pieces):
  board = [[0] * 8 for i in range(8)]

  num_rolls = num_pieces - 2
  maxm = len(rolls) - 1
  minm = 0

  # occupied board spaces
  occupied = []
  assign_kings(occupied, board)

  start = random.randint(0, 1)

  side = 1 if start == 0 else -1

  i = 0
  while i < num_rolls:

    num = roll_range(minm, maxm)
    
    piece = rolls[num]
    if piece == 'q':
      maxm -= 1

    opposite_balance = balance_dict[piece]
    assign_piece(occupied, piece, side, board)

    i += 1

    if i >= num_rolls:
      return board

    # assign opposite pieces
    choice = random.choice(opposite_balance)
    if len(choice) != 1 and i + len(choice) > num_rolls:
      choice = random.choice(opposite_balance)

    for opposite_piece in choice:

      assign_piece(occupied, opposite_piece, (side * -1), board)
      i += 1

      if i >= num_rolls:
        return board

  return board


def assign_piece(occupied, piece, side, board):

    piece = piece.upper() if side == 1 else piece

    pos, row, col = rand_board_idx(occupied)

    while ((piece == 'p' or piece == 'P') and (row == 0 or row == 7)):
      pos, row, col = rand_board_idx(occupied)

    occupied.append(pos)
    board[row][col] = piece

    return


def assign_kings(occupied, board):
  pos, row, col = rand_board_idx(occupied)
  occupied.append(pos)
  board[row][col] = 'k'

  pos, row2, col2 = rand_board_idx(occupied)
  while adjacent_coords(row, col, row2, col2):
    pos, row2, col2 = rand_board_idx(occupied)
 
  occupied.append(pos)
  board[row2][col2] = 'K'


def adjacent_coords(row, col, row2, col2):
  return (
    ((row == row2) and (col == col2 + 1 or col == col2 - 1)) or
    ((row == row2 + 1) and (col == col2 + 1)) or
    ((row == row2 - 1) and (col == col2 - 1)) or
    ((row == row2 - 1) and (col == col2 + 1)) or
    ((row == row2 + 1) and (col == col2 - 1)) or
    ((col == col2) and (row == row2 + 1 or row == row2 - 1))
    )


def rand_board_idx(occupied):
  
  piece_pos = random.randint(0, 63)

  while piece_pos in occupied:
    piece_pos = random.randint(0, 63)

  row = piece_pos // 8
  col = piece_pos % 8


  return piece_pos, row, col


def roll_range(minm, maxm):
  return random.randint(minm, maxm)



def random_board(max_depth=200):
	board = chess.Board()
	depth = random.randrange(0, max_depth)

	for _ in range(depth):
		all_moves = list(board.legal_moves)
		random_move = random.choice(all_moves)
		board.push(random_move)
		if board.is_game_over():
			break

	return board  


if __name__ == '__main__':
  main(sys.argv[1:])

