# This file is part of NPFL139 <http://github.com/ufal/npfl139/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import numpy as np

from .az_quiz import AZQuiz
from .board_game_player import BoardGamePlayer


# Utils
CENTER = 12
CORNER_STONES = [16, 19]
LINE_3 = [11, 12, 13]
CURVE_3 = [11, 12, 18]
BOTTOM_LINE = [16, 17, 18, 19]

def intersect(test1, test2):
    return list(set(test1).intersection(set(test2)))

def union(test1, test2):
    return list(set(test1).union(set(test2)))

# Rotations
BOARD_ROTATIONS = np.array([
    [ 0,  0, 21, 21, 27, 27],
    [ 1,  2, 22, 15, 20, 26],
    [ 2,  1, 15, 22, 26, 20],
    [ 3,  5, 23, 10, 14, 25],
    [ 4,  4, 16, 16, 19, 19],
    [ 5,  3, 10, 23, 25, 14],
    [ 6,  9, 24,  6,  9, 24],
    [ 7,  8, 17, 11, 13, 18],
    [ 8,  7, 11, 17, 18, 13],
    [ 9,  6,  6, 24, 24,  9],
    [10, 14, 25,  3,  5, 23],
    [11, 13, 18,  7,  8, 17],
    [12, 12, 12, 12, 12, 12],
    [13, 11,  7, 18, 17,  8],
    [14, 10,  3, 25, 23,  5],
    [15, 20, 26,  1,  2, 22],
    [16, 19, 19,  4,  4, 16],
    [17, 18, 13,  8,  7, 11],
    [18, 17,  8, 13, 11,  7],
    [19, 16,  4, 19, 16,  4],
    [20, 15,  1, 26, 22,  2],
    [21, 27, 27,  0,  0, 21],
    [22, 26, 20,  2,  1, 15],
    [23, 25, 14,  5,  3, 10],
    [24, 24,  9,  9,  6,  6],
    [25, 23,  5, 14, 10,  3],
    [26, 22,  2, 20, 15,  1],
    [27, 21,  0, 27, 21,  0],
])

INVERSE_ROTATION = [0, 1, 4, 3, 2, 5]

def rotate_list(l, r):
    s = np.zeros(28)
    s[l] = 1
    rs = rotate_situation(s, r)
    return rs.nonzero()[0]

def rotate_situation(situation, r=0):
    rotation = BOARD_ROTATIONS[:, r]
    rotated_situation = np.zeros(len(situation))
    for i in range(len(situation)):
        rotated_situation[i] = situation[rotation[i]]
    return rotated_situation

def all_variants(func):
    def inner(my, enemy, allowed, generator):
        rotations = INVERSE_ROTATION.copy()
        generator.shuffle(rotations)
        possible_moves = []
        for rotation in rotations:
            rotated_my = rotate_list(my, rotation)
            rotated_enemy = rotate_list(enemy, rotation)
            rotated_allowed = rotate_list(allowed, rotation)
            action = func(rotated_my, rotated_enemy, rotated_allowed, generator)
            if action>-1:
                action = BOARD_ROTATIONS[action, rotation]
            if action>=0:
                possible_moves.append(action)
        if len(possible_moves)>0:
            m = intersect(possible_moves, allowed)
            m = generator.choice(m)
        else:
            m=-1
        return m
    return inner

# Rules
def choose_random(arr, allowed, generator):
    if len(intersect(arr,allowed))==0:
        return -1
    return generator.choice(intersect(arr,allowed))

def vidlicky(enemy, allowed, pole, generator):
    if len(intersect(enemy, pole))>=1:
        return choose_random(pole, allowed, generator)
    return -1

@all_variants
def priprav_vidle(my, enemy, allowed, generator):
    if len(intersect(my, [13,12,7]))==3:
        if len(intersect(enemy, [17,18,19]))==3:
            if len(my)==3 and len(enemy)==3:
                return 10
    return -1

@all_variants
def obchazi_vidle_5(my, enemy, allowed, generator):
    if len(intersect(my, [13,12,7, 10]))==4:
        if len(intersect(enemy, [17,18,19]))==3:
            if 11 in enemy and 6 in allowed: return 6
            if 6 in enemy and 11 in allowed: return 11
        if len(intersect(enemy, BOTTOM_LINE))==4 and 15 in allowed: return 15
    if len(intersect(my, [13,12,7, 10, 15]))==5:
        if len(intersect(enemy, BOTTOM_LINE))==4:
            if 22 in enemy and 21 in allowed: return 21
            if 21 in enemy and 22 in allowed: return 22
    if len(intersect(my,[ 8, 11, 12])) ==3:
         if len(intersect(enemy,[13, 17, 18]))==3:
             if 16 in allowed: return 16
    if len(intersect(my,[ 8, 11, 12, 16])) ==4:
         if len(intersect(enemy,[13, 17, 18, 23]))==4:
             if 22 in allowed: return 22

    return -1

@all_variants
def obsad_prostredek(my, enemy, allowed, generator):
    if len(intersect(my, LINE_3))>=1 and len(intersect(enemy, LINE_3))==0:
        return choose_random(LINE_3, allowed, generator)
    return -1

@all_variants
def prostredni_cara(my, enemy, allowed, generator):
    if len(intersect(my, LINE_3))>=2 and len(intersect(enemy, LINE_3))==0:
        return choose_random(LINE_3, allowed, generator)
    return -1

@all_variants
def prostredni_krivka(my, enemy, allowed, generator):
    if len(intersect(my, [11,12]))==2 and 13 in enemy:
        return choose_random([8,18], allowed, generator)
    return -1

@all_variants
def obchazi_vidle_1(my, enemy, allowed, generator):
    if len(intersect(my, LINE_3))==3:
        return vidlicky(enemy, allowed, [10,6], generator)
    return -1

@all_variants
def obchazi_vidle_2(my, enemy, allowed, generator):
    if len(intersect(my, CURVE_3))==3:
        return vidlicky(enemy, allowed, [24,25], generator)
    return -1

@all_variants
def obchazi_vidle_3(my, enemy, allowed, generator):
    if len(intersect(my, CURVE_3))==3:
        return vidlicky(enemy, allowed, [10,6], generator)
    return -1

@all_variants
def obchazi_vidle_4(my, enemy, allowed, generator):
    if 16 in my:
        return vidlicky(enemy, allowed, [16], generator)
    return -1

@all_variants
def obchazi_vidle_6(my, enemy, allowed, generator):
    if len(intersect(my, [7,8,13,18]))==4:
        ret = -1

        if ret==-1: ret = vidlicky(enemy, allowed, [3,6], generator)
        if ret==-1: ret = vidlicky(enemy, allowed, [5,9], generator)
        if ret==-1: ret = vidlicky(enemy, allowed, [24,25], generator)
        return ret

    return -1

@all_variants
def pozor_dole(my, enemy, allowed, generator):
    if len(intersect(my, LINE_3))==3 and len(intersect(enemy, BOTTOM_LINE))>=3:
        return choose_random(BOTTOM_LINE, allowed, generator)
    return -1

@all_variants
def pribliz_se_ke_stedu(my, enemy, allowed, generator):
    if len(intersect(my, LINE_3))==3 and len(intersect(my, [17,18]))==0:
        if 17 in allowed: return 17
        if 18 in allowed: return 18
    return -1

@all_variants
def spoj_stranu_stredem(my, enemy, allowed, generator):
    if len(intersect(my, BOTTOM_LINE))>0:
        for field in intersect(my, BOTTOM_LINE):
            if field+6 in allowed: return field+6
            if field+7 in allowed: return field+7
    return -1

@all_variants
def spoj_stranu_rohem(my, enemy, allowed, generator):
    if len(intersect(my, [15,22]))>0:
        if 21 in allowed: return 21
    return -1

@all_variants
def corner_stone(my, enemy, allowed, generator):
    if len(intersect(my, CORNER_STONES))==0:
        return choose_random(CORNER_STONES, allowed, generator)
    return -1

@all_variants
def corner_stones(my, enemy, allowed, generator):
    if len(intersect(enemy, [16, 19]))==2:
        if 4 in allowed: return 4
    return -1

@all_variants
def dolni_cara(my, enemy, allowed, generator):
    if len(intersect(my, CORNER_STONES))==2:
        if len(intersect(enemy, BOTTOM_LINE))==0:
            return choose_random(BOTTOM_LINE, allowed, generator)
    return -1

@all_variants
def sikma_cara(my, enemy, allowed, generator):
    if len(intersect(enemy, [18,13]))==0:
        return choose_random([18,13], allowed, generator)
    if len(intersect(enemy, [18,13]))==2:
        if 8 in allowed: return 8
    if len(intersect(enemy, [18,13,8]))==3:
        if 4 in allowed: return 4
        if 7 in allowed: return 7
        if 5 in allowed: return 5
    if len(intersect(enemy, [18,13,8,5]))==4:
        if 2 in allowed: return 2
        if 2 in my and 21 in allowed: return 21
    return -1

def hraj_nahodne(my, enemy, allowed, generator):
    return choose_random(allowed, allowed, generator)

@all_variants
def prodluz_dolni_caru(my, enemy, allowed, generator):
    if len(intersect(my, BOTTOM_LINE))==4:
        if len(intersect(enemy, [14,20]))==1:
            return choose_random([14,20], allowed, generator)
        if len(intersect(enemy, [15,10]))==1:
            return choose_random([15,10], allowed, generator)
    return -1

@all_variants
def utok_bez_duvodu(my, enemy, allowed, generator):
    if len(intersect(my, [7,11,8,17]))==4:
        if len(intersect(allowed, [6,3,5,9,23,24]))==6:
            return choose_random([6,3,5,9,23,24], allowed, generator)
    return -1

@all_variants
def lepsi_pulkolecko(my, enemy, allowed, generator):
    if len(intersect(union(allowed,my), [11,17,7,8]))==4 and len(intersect(union(allowed,enemy), [18,13]))==2:
        return choose_random([11,17,7,8], allowed, generator)
    return -1

chain_mam_stred = [
    utok_bez_duvodu,
    obchazi_vidle_1,
    obchazi_vidle_2,
    obchazi_vidle_3,
    obchazi_vidle_4,
    obchazi_vidle_5,
    priprav_vidle,
    pozor_dole,
    prostredni_krivka,
    pribliz_se_ke_stedu,
    prostredni_cara,
    spoj_stranu_stredem,
    spoj_stranu_rohem,
    obsad_prostredek,
    hraj_nahodne,
]

chain_nemam_stred = [
    obchazi_vidle_6,
    utok_bez_duvodu,
    lepsi_pulkolecko,
    corner_stone,
    corner_stones,
    prodluz_dolni_caru,
    sikma_cara,
    dolni_cara,
    spoj_stranu_stredem,
    spoj_stranu_rohem,
    hraj_nahodne,
]

def apply_rules(rules, my, enemy, allowed, generator):
    action = -1
    for rule in rules:
        if action==-1:
            action = rule(my, enemy, allowed, generator)
    return action


class AZQuizPlayerForkHeuristic(BoardGamePlayer[AZQuiz]):
    """An elaborate heuristic for AZQuiz."""
    def __init__(self, seed: int | None = None):
        self._generator = np.random.RandomState(seed)

    def play(self, game: AZQuiz) -> int:
        board = game.board_features
        situation = np.array([board[y, x] for y in range(7) for x in range(y + 1)])
        my = situation[:, game.to_play].nonzero()[0]
        enemy = situation[:, 1 - game.to_play].nonzero()[0]
        taken = union(my, enemy)
        allowed = [action for action in range(28) if action not in taken]
        if CENTER in allowed:
            return CENTER
        if CENTER in enemy:
            return apply_rules(chain_nemam_stred, my, enemy, allowed, self._generator)
        if CENTER in my:
            return apply_rules(chain_mam_stred, my, enemy, allowed, self._generator)


AZQuiz.register_player("fork_heuristic", AZQuizPlayerForkHeuristic)
