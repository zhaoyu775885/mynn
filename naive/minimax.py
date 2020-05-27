# -*- coding: utf-8 -*-

import numpy as np
import random

class Board():
    def __init__(self):
        self.board = -np.ones([3, 3], dtype=np.int8)
        self.blank_val = -1
        
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
        
        def role_win(role):
            for row in range(3):
                if row_win(row, role): return True
            for col in range(3):
                if col_win(col, role): return True
            if main_diag_win(role): return True
            if sub_diag_win(role): return True
            return False
            
        if role_win(0): return 0
        if role_win(1): return 1
        
        return self.blank_val
    
    def over(self):
        winner = self.winner()
        if winner != self.blank_val:
            return True
        return not (self.board==self.blank_val).any()
    
    def avlb_positions(self):
        blank_positions = []
        for i in range(3):
            for j in range(3):
                if self.board[i, j]==self.blank_val:
                    blank_positions.append(3*i+j)
#        blank_positions.reverse()
        random.shuffle(blank_positions)
        return blank_positions
    
    def place_role(self, pos, role):
        row, col = pos//3, pos%3
        self.board[row, col] = role
        
    def clear_place(self, pos):
        row, col = pos//3, pos%3
        self.board[row, col] = self.blank_val

    def evaluate(self, role):
        def row_score(row, role):
            exist_role = (self.board[row,:]==role).any()
            exist_oppo = (self.board[row,:]==1-role).any()
            if exist_oppo == exist_role:
                return 0
            if exist_role:
                role_count = np.count_nonzero(self.board[row,:]==role)
                return role_count if role_count<3 else 100
            if exist_oppo:
                role_count = np.count_nonzero(self.board[row,:]==1-role)
                return -2 if role_count==1 else -100
#                return -np.count_nonzero(self.board[row, :]==1-role)-1            
            
        def col_score(col, role):
            exist_role = (self.board[:,col]==role).any()
            exist_oppo = (self.board[:,col]==1-role).any()
            if exist_oppo == exist_role:
                return 0
            if exist_role:
                role_count = np.count_nonzero(self.board[:, col]==role)
                return role_count if role_count<3 else 100
            if exist_oppo:
                role_count = np.count_nonzero(self.board[:, col]==1-role)
                return -2 if role_count==1 else -100
        
        def main_diag_score(role):
            diag = np.diag(self.board)
            exist_role = (diag==role).any()
            exist_oppo = (diag==1-role).any()
            if exist_oppo == exist_role:
                return 0
            if exist_role:
                role_count = np.count_nonzero(diag==role)
                return role_count if role_count<3 else 100
            if exist_oppo:
                role_count = np.count_nonzero(diag==1-role)
                return -2 if role_count==1 else -100
        
        def sub_diag_score(role):
            sub_diag = np.array([self.board[x,2-x] for x in range(3)])
            exist_role = (sub_diag==role).any()
            exist_oppo = (sub_diag==1-role).any()
            if exist_oppo == exist_role:
                return 0
            if exist_role:
                role_count = np.count_nonzero(sub_diag==role)
                return role_count if role_count<3 else 100
            if exist_oppo:
                role_count = np.count_nonzero(sub_diag==1-role)
                return -2 if role_count==1 else -100
        
        score = 0
        for row in range(3):
            score += row_score(row, role)
        for col in range(3):
            score += col_score(col, role)
        score += main_diag_score(role)
        score += sub_diag_score(role)
        return score
    
    def random_play(self, role=0):
        if self.over():
            return
        blank_positions = self.avlb_positions()
        pos = blank_positions[np.random.randint(0, len(blank_positions))]
        self.place_role(pos, role)
        
    def load_board(self, info):
        self.board = info
