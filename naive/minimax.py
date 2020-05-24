# -*- coding: utf-8 -*-

import numpy as np
import copy

def minimax(board, role):
    blank_positions = board.avlb_positions()
    act_val = -np.inf
    act = -1
    for pos in blank_positions:
        board.place_role(pos, role)
        val = min_value(board, role)
        if val > act_val:
            act_val = val
            act = pos
        board.clear_place(pos)
    return act

def min_value(board, role):
    if board.over():
        winner = board.winner
        if winner == 0:
            return 0
        elif winner == role:
            return 1
        else:
            return -1
    blank_positions = board.avlb_positions()
    act_val = np.inf
    act = -1
    for pos in blank_positions:
        board.place_role(pos, 3-role)
        val = max_value(board, role)
        if val < act_val:
            act_val = val
            act = pos
        board.clear_place(pos)
    return act_val

def max_value(board, role):
    if board.over():
        winner = board.winner
        if winner == 0:
            return 0
        elif winner == role:
            return 1
        else:
            return -1
    blank_positions = board.avlb_positions()
    act_val = -np.inf
    act = -1
    for pos in blank_positions:
        board.place_role(pos, role)
        val = min_value(board, role)
        if val > act_val:
            act_val = val
            act = pos
        board.clear_place(pos)
    return act_val

def strategy(board, role):
    # for simulation on virtual board
    vboard = copy.deepcopy(board)
    
    # minimax search
    pos = minimax(vboard, role)
    print(pos)
    
    row, col = pos//3, pos%3
    
    # take action
    board.board[row, col] = role

def play(board):
    while(not board.over()):
        # player 1: minimax search
#        board.random_play(1)
        strategy(board, 1)
        
        # player 2: random_play
        board.random_play(2)
        
        print(board, '\n--------------------')
        
def avlb_positions(board):
    blank_positions = []
    for i in range(3):
        for j in range(3):
            if board[i, j]==0:
                blank_positions.append(3*i+j)
    return blank_positions

class Board():
    def __init__(self):
        self.board = np.zeros([3, 3], dtype=np.int8)
        
    def __str__(self):
        return '{0}'.format(self.board)
    
    def winner(self):
        def row_win(row, role):
            for col in range(3):
                if self.board[row, col] != role:
                    return False
            return True
        
        def col_win(col, role):
            for row in range(3):
                if self.board[row, col] != role:
                    return False
            return True
        
        def main_diag_win(role):
            for _ in range(3):
                if self.board[_, _] != role:
                    return False
            return True
        
        def sub_diag_win(role):
            for _ in range(3):
                if self.board[_, 2-_] != role:
                    return False
            return True
                    
        for row in range(3):
            if row_win(row, 1): return 1
            if row_win(row, 2): return 2
            
        for col in range(3):
            if col_win(col, 1): return 1
            if col_win(col, 2): return 2
        
        if main_diag_win(1): return 1
        if main_diag_win(2): return 2
        if sub_diag_win(1): return 1
        if sub_diag_win(2): return 2
        
        return 0
    
    def over(self):
        winner = self.winner()
        if winner != 0:
            return True           
        return not (self.board==0).any()
    
    def avlb_positions(self):
        blank_positions = []
        for i in range(3):
            for j in range(3):
                if self.board[i, j]==0:
                    blank_positions.append(3*i+j)
        return blank_positions
    
    def place_role(self, pos, role):
        row, col = pos//3, pos%3
        self.board[row, col] = role
        
    def clear_place(self, pos):
        row, col = pos//3, pos%3
        self.board[row, col] = 0
    
    def random_play(self, role=1):
        if self.over():
            return
        blank_positions = self.avlb_positions()
        
        pos = blank_positions[np.random.randint(0, len(blank_positions))]
        
        self.place_role(pos, role)
        
    def load_board(self, info):
        self.board = info
        
if __name__ == '__main__':
    board = Board()
    print(board)
    info = np.array([[1, 1, 2],
                     [1, 0, 0],
                     [2, 0, 2]])
    print('==============================')
    play(board)
    
    
    