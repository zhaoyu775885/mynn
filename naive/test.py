# -*- coding: utf-8 -*-

import numpy as np
import copy

from Board import Board

HUMAN_PLAYER = 0
ROBOT_PALYER = 1

def min_max_value(board, act_role, pln_role, depth):
    r'''
    re-implement the recursive part of minimax algorithm in conciser way
    '''
    if board.over() or depth<0:
        return board.evaluate(act_role)
    
    act_val = -np.inf if act_role==pln_role else np.inf
    blank_positions = board.avlb_positions()
    for pos in blank_positions:
        board.place_role(pos, pln_role)
        val = min_max_value(board, act_role, 1-pln_role, depth-1)
        if act_role == pln_role:
            if val > act_val:
                act_val = val
        else:
            if val < act_val:
                act_val = val
        board.clear_place(pos)
    
    return act_val

def minimax_v2(board, role, depth=4):
    blank_positions = board.avlb_positions()
    act_val = -np.inf
    act = -1
    print(blank_positions)
    for pos in blank_positions:
        board.place_role(pos, role)
        val = min_max_value(board, role, 1-role, depth-1)
#        print('f: ', val)
        if val > act_val:
            act_val = val
            act = pos
        board.clear_place(pos)
    return act

def minimax(board, role, depth=4):
    blank_positions = board.avlb_positions()
    act_val = -np.inf
    act = -1
    print(blank_positions)
    for pos in blank_positions:
        board.place_role(pos, role)
        val = min_value(board, role, depth-1)
#        print('f: ', val)
        if val > act_val:
            act_val = val
            act = pos
        board.clear_place(pos)
    return act

def min_value(board, role, depth):
    if board.over() or depth<0:
        return board.evaluate(role)
    blank_positions = board.avlb_positions()
    act_val = np.inf
    for pos in blank_positions:
        board.place_role(pos, 1-role)
        val = max_value(board, role, depth-1)
        if val < act_val:
            act_val = val
        board.clear_place(pos)
    return act_val

def max_value(board, role, depth):
    if board.over() or depth<0:
        return board.evaluate(role)
    blank_positions = board.avlb_positions()
    act_val = -np.inf
    for pos in blank_positions:
        board.place_role(pos, role)
        val = min_value(board, role, depth-1)
        if val > act_val:
            act_val = val
        board.clear_place(pos)
    return act_val

def strategy(board, role):
    # simulation on virtual board
    
    if not board.over():
        vboard = copy.deepcopy(board)
        pos = minimax_v2(vboard, role, depth=4)
#        pos = minimax(vboard, role)
#        print(pos)
        board.place_role(pos, role)

def play(board : Board , P1_AI=True, P2_AI=True):
    while(not board.over()):
        if P1_AI:
            strategy(board, HUMAN_PLAYER)
        else:
            board.random_play(HUMAN_PLAYER)
        
        if P2_AI:
            strategy(board, ROBOT_PALYER)            
        else:
            board.random_play(ROBOT_PALYER)

        print(board, '\n--------------------')

    winner = board.winner()
    if winner == -1:
        print('Draw game')
    else:
        print('Winner is Player {0}'.format(winner+1))


def debug():
    board = Board()
    print(board, '\n==============================')
    info = np.array([[1, 0, -1],
                     [-1, 1, -1],
                     [0, -1, -1]])
    board.load_board(info)
    print(board, '\n==============================')
    role = 0
    pos = 3
    act_role = role
    pln_role = role
    board.place_role(pos, pln_role)
    act, val = minimax_v2(board, act_role, 1-pln_role, 3)
    print(act, val)
    
def main():
    board = Board()
    print(board, '\n==============================')
    play(board, P1_AI=True, P2_AI=True)
    
if __name__ == '__main__':
#    debug()
    main()
