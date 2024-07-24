import numpy as np
import random

# Tic-Tac-Toe game board
board = np.zeros((3, 3))

# Function to check if a player has won
def check_win(board, player):
    for i in range(3):
        if board[i, 0] == board[i, 1] == board[i, 2] == player:
            return True
        if board[0, i] == board[1, i] == board[2, i] == player:
            return True
    if board[0, 0] == board[1, 1] == board[2, 2] == player:
        return True
    if board[0, 2] == board[1, 1] == board[2, 0] == player:
        return True
    return False

# Minimax algorithm with alpha-beta pruning
def minimax(board, depth, alpha, beta, player):
    if check_win(board, player):
        return -10 + depth
    if check_win(board, -player):
        return 10 - depth
    if depth == 0:
        return 0

    if player == 1:
        value = -np.inf
        for i in range(3):
            for j in range(3):
                if board[i, j] == 0:
                    board[i, j] = player
                    value = max(value, minimax(board, depth-1, alpha, beta, -player))
                    board[i, j] = 0
                    alpha = max(alpha, value)
                    if alpha >= beta:
                        return value
        return value
    else:
        value = np.inf
        for i in range(3):
            for j in range(3):
                if board[i, j] == 0:
                    board[i, j] = player
                    value = min(value, minimax(board, depth-1, alpha, beta, -player))
                    board[i, j] = 0
                    beta = min(beta, value)
                    if beta <= alpha:
                        return value
        return value

# GPU 1: Evaluation function that prioritizes controlling the center
def eval_func_1(board):
    center_control = 0
    if board[1, 1] == 1:
        center_control = 1
    return minimax(board, 5, -np.inf, np.inf, 1) + center_control

# GPU 2: Evaluation function that prioritizes blocking the opponent
def eval_func_2(board):
    block_opponent = 0
    for i in range(3):
        for j in range(3):
            if board[i, j] == -1:
                block_opponent += 1
    return minimax(board, 5, -np.inf, np.inf, -1) + block_opponent

# Main game loop
while True:
    # GPU 1's turn
    best_move = None
    best_score = -np.inf
    for i in range(3):
        for j in range(3):
            if board[i, j] == 0:
                board[i, j] = 1
                score = eval_func_1(board)
                board[i, j] = 0
                if score > best_score:
                    best_score = score
                    best_move = (i, j)
    board[best_move[0], best_move[1]] = 1

    # Check if GPU 1 has won
    if check_win(board, 1):
        print("GPU 1 wins!")
        break

    # GPU 2's turn
    best_move = None
    best_score = np.inf
    for i in range(3):
        for j in range(3):
            if board[i, j] == 0:
                board[i, j] = -1
                score = eval_func_2(board)
                board[i, j] = 0
                if score < best_score:
                    best_score = score
                    best_move = (i, j)
    board[best_move[0], best_move[1]] = -1

    # Check if GPU 2 has won
    if check_win(board, -1):
        print("GPU 2 wins!")
        break

    # Print the current board state
    print(board)