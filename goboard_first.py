### Very slow object oriented go board implementation for practice
from collections import namedtuple
import numpy as np
import random
import enum
from copy import deepcopy
import time
import math

DEFAULT_BOARD_SIZE = 5 
BLACK, WHITE, EMPTY = 'O', 'X', '.'
KOMI = 0

def swap_colors(color):
    if color == BLACK:
        return WHITE
    elif color == WHITE:
        return BLACK
    else:
        return color
## Japanese coordinate convention
## the upper side star point from left to right are
## 4-4, 10,4, 16,4

## c are coordinates in (n,m) form, reduced by 1 since arrays start at 0
## So the 3-4 point would be (2,3)
## flat coordinates are preceded by an f
def flat_cord(size,c):
    return size * c[0] + c[1]

def unflat_cord(size, fc):
    return divmod(fc, size)
# check if a move is possible
def is_onboard(size, c):
    return c[0] % size == c[0] and c[1] % size == c[1]

def get_valid_neighbors(size, fc):
    x, y = unflat_cord(size, fc)
    possible_neighbors = ((x+1,y),(x-1,y),(x, y+1), (x, y-1))
    return [flat_cord(size, n) for n in possible_neighbors if is_onboard(size, n)]

# maybe skip this class
class Player(enum.Enum):
    black = 1
    white = 2

    @property
    def other(self):
        return Player.black if self == Player.white else Player.white

class Move():
    def __init__(self, fcord = None, is_pass=False, is_resign=False):
        assert (fcord is not None) ^ is_pass ^ is_resign
        self.fcord = fcord
        self.is_play = (self.fcord is not None)
        self.is_pass = is_pass
        self.is_resign = is_resign

    @classmethod
    def play(cls, fcord):
        return Move(fcord = fcord)

    @classmethod
    def resign(cls):
        return Move(is_resign=True)
    @classmethod
    def pass_turn(cls):
        return Move(is_pass=True)

#Maybe encode neighbors with the board to speed up calculations?
class Board:
    def __init__(self, size=DEFAULT_BOARD_SIZE, state=(DEFAULT_BOARD_SIZE*DEFAULT_BOARD_SIZE)* EMPTY):
        self.size = size
        self.state = state
        self.neighbors = [get_valid_neighbors(size, fc) for fc in range(size*size)]
# chain consists of all reachable point of the same color (or empty)
# reached consists of the chain's neigborhood
    def find_reached(self, fc):
        color = self.state[fc]
        chain = set([fc])
        reached = set()
        frontier = [fc]
        while frontier:
            current_fc = frontier.pop()
            chain.add(current_fc)
            for fn in self.neighbors[current_fc]:
                if self.state[fn] == color and not fn in chain:
                    frontier.append(fn)
                elif self.state[fn] != color:
                    reached.add(fn)
        return chain, reached

    def place_stone(self, color, fc):
        return Board(self.size, self.state[:fc] + color + self.state[fc+1:])

    def output_as_matrix(self):
        mat = np.array(list(self.state))
        return mat.reshape(self.size,self.size)

    # helper function
    def bulk_place_stones(self, color, stones):
        byteboard = bytearray(self.state, encoding='ascii')
        color = ord(color)
        for fstone in stones:
            byteboard[fstone] = color
        return byteboard.decode('ascii')

    # check liberties of group containing this stone
    def maybe_capture_stones(self, fc):
        chain, reached = self.find_reached(fc)
        if not any(self.state[fr] == EMPTY for fr in reached):
            bord = self.bulk_place_stones(EMPTY, chain)
            return bord, chain
        else:
            return self.state, []

    def play_move_incomplete(self, fc, color):
        if self.state[fc] != EMPTY:
            raise IllegalMove
        self.state = place_stone(self, color, fc).state

        opp_color = swap_colors(color)
        opp_stones = []
        my_stones = []
        for fn in self.neighbors[fc]:
            if self.state[fn]==color:
                my_stones.append(fn)
            elif self.state[fn] == opp_color:
                opp_stones.append(fn)

        for fs in opp_stones:
            self.state, _ = maybe_capture_stones(self, fs)

        for fs in my_stones:
            self.state, _ = maybe_capture_stones(self, fs)

        return self.state


    def is_koish(self, fc):
        'Check if fc is surrounded on all sides by 1 color, and return that color'
        if self.state[fc] != EMPTY: return None
        neighbor_colors = {self.state[fn] for fn in self.neighbors[fc]}
        if len(neighbor_colors) == 1 and not EMPTY in neighbor_colors:
            return list(neighbor_colors)[0]
        else:
            return None

    def is_eye(self, fc, color):
        if self.state[fc]!=EMPTY:
            return False
        for n in self.neighbors[fc]:
            if self.state[n] != color:
                return False
        x, y = unflat_cord(self.size, fc)
        friendly_corners = 0
        off_board_corners = 0
        corners = [
            (x-1, y-1),
            (x-1, y+1),
            (x+1, y-1),
            (x+1, y+1)
        ]
        for corner in corners:
            if is_onboard(self.size, corner):
                corner_color = self.state[flat_cord(self.size, corner)]
                if corner_color == color:
                    friendly_corners += 1
            else:
                off_board_corners += 1
        if off_board_corners >0:
            return (off_board_corners + friendly_corners) == 4
        return friendly_corners >= 3
            
class IllegalMove(Exception): pass

class Position(namedtuple('Position', ['board', 'ko'])):
    @staticmethod
    def initial_state():
        return Position(board=Board(), ko=None)

#    def __init__(self, board, ko):
#        self.board
    def get_board(self):
        return self.board
    def get_ko(self):
        return self.ko

    def __str__(self):
        import textwrap
        return '\n'.join(textwrap.wrap(self.board.state, self.board.size))

    #weirdly simplistic
    def is_valid_move(self, fc, color):
        board, ko = self
        if fc == ko:
            return False
        if board.state[fc] !=EMPTY:
            return False

        possible_ko_color = board.is_koish(fc)
        bord = board.place_stone(color, fc)

        opp_color = swap_colors(color)
        opp_stones = []
        my_stones = []
        for fn in bord.neighbors[fc]:
            if bord.state[fn] == color:
                my_stones.append(fn)
            elif bord.state[fn] == opp_color:
                opp_stones.append(fn)

        opp_captured = 0
        for fs in opp_stones:
            bord.state, captured = bord.maybe_capture_stones(fs)
            opp_captured += len(captured)

        # Check for suicide
        bord.state, captured = bord.maybe_capture_stones(fc)
        if captured:
            return False
        return True

    def play_move(self, fc, color):
        board, ko = self
        if fc == ko:
            raise IllegalMove("%s\n Move at %s illegally retakes ko." % (self, fc))

        if board.state[fc] != EMPTY:
            raise IllegalMove("%s\n Stone exists at %s." % (self, fc))

        possible_kos = []
        possible_ko_color = board.is_koish(fc)
        board = board.place_stone(color, fc)

        opp_color = swap_colors(color)
        opp_stones = []
        my_stones = []
        for fn in board.neighbors[fc]:
            if board.state[fn] == color:
                my_stones.append(fn)
            elif board.state[fn] == opp_color:
                opp_stones.append(fn)

        opp_captured = 0
        possible_ko = None
        for fs in opp_stones:
            board.state, captured = board.maybe_capture_stones(fs)
            opp_captured += len(captured)
            possible_kos = possible_kos + list(captured)
        if opp_captured == 1 and possible_ko_color == opp_color:
            new_ko = possible_kos[0]
        else:
            new_ko = None
        # Check for suicide
        board.state, captured = board.maybe_capture_stones(fc)
        if captured:
            raise IllegalMove("\n%s\n Move at %s is suicide." % (self, fc))
        return Position(board, new_ko)
    def hum_cord_play_move(self, c, color):
        x, y = c[0], c[1]
        return self.play_move(flat_cord(self.board.size, (x-1,y-1)),color)

    def score(self):
        board = Board(self.board.size,self.board.state)
        while EMPTY in board.state:
            fempty = board.state.index(EMPTY)
            empties, borders = board.find_reached(fempty)
            if not borders:
                return 0
            possible_border_color = board.state[list(borders)[0]]
            if all(board.state[fb] == possible_border_color for fb in borders):
                board.state = board.bulk_place_stones(possible_border_color, empties)
            else:
                # if an empty intersection reaches both white and black,
                # then it belongs to neither player. 
                board.state = board.bulk_place_stones('?', empties)
        return board.state.count(BLACK) - board.state.count(WHITE)

    def get_liberties(self):
        board = self.board
        liberties = bytearray(board.size*board.size)
        for color in (WHITE, BLACK):
            while color in board.state:
                fc = board.state.index(color)
                stones, borders = board.find_reached(fc)
                num_libs = len([fb for fb in borders if board.state[fb] == EMPTY])
                for fs in stones:
                    liberties[fs] = num_libs
                board.state = board.bulk_place_stones('?', stones)
        return list(liberties)

    def print_position(self):
        print('\n'.join(''.join(str(col)+' ' for col in row) for row in self.board.output_as_matrix()))


class Gamestate(namedtuple('Gamestate', ['player', 'position','prev_state','last_move'])):
    @staticmethod
    def initial_state():
        return Gamestate(player = BLACK, position = Position(Board(), None),prev_state=None, last_move=None)
    def apply_move(self, move):
        if move.is_play:
            next_position = self.position.play_move(move.fcord,self.player)
        else:
            next_position = self.position
        return Gamestate(swap_colors(self.player),next_position, self, move)
    def legal_moves(self):
        candidates =[]
        for fn in range(pow(self.position.board.size,2)):
            if self.position.is_valid_move(fn, self.player):
                candidates.append(Move.play(fn))
        candidates.append(Move.pass_turn())
        candidates.append(Move.resign())
        return candidates

    def count_score(self):
        return self.position.score()

    def is_over(self):
        if self.last_move == None:
            return False
        if self.last_move.is_resign:
            return True
        second_last_move = self.prev_state.last_move
        if second_last_move == None:
            return False
        return self.last_move.is_pass and second_last_move.is_pass

    def winner(self):
        if not self.is_over():
            return None
        if self.last_move.is_resign:
            return self.player
        result = self.count_score()
        if result > KOMI:
            return BLACK
        return WHITE

class Agent:
    def __init__(self):
        pass

    def select_move(self, game_state):
        raise NotImplementedError()

class RandomBot(Agent):
    def select_move(self, gamestate):
        """Random move besides not filling your own eyes"""
        candidates = []
        for fr in range(gamestate.position.board.size*gamestate.position.board.size):
            if gamestate.position.is_valid_move(fr,gamestate.player) and not gamestate.position.board.is_eye(fr, gamestate.player):
                candidates.append(fr)
        if not candidates:
            return Move.pass_turn()
        return Move.play(random.choice(candidates))

def TestKo():
    B = Board()
    B.state ='XO........X.......X..............................................................'
    P = Position(B, 9)
    G = Gamestate(BLACK, P)
    M = Move.play(9)
    G.apply_move(M)


def simulate_random_game_from_gamestate(gamestate):
    G = gamestate
    r = RandomBot()
    pass_count = 0
    next_move = r.select_move(G.position, G.player)
    while pass_count < 2:
        if next_move.is_pass:
            pass_count += 1
        else:
            pass_count = 0
        G = G.apply_move(next_move)
        next_move = r.select_move(G.position, G.player)
    return G
# Simply game evaluation function counting the number of stones on the board
def capture_diff(gamestate):
    black_stones = gamestate.position.board.state.count(BLACK)
    white_stones = gamestate.position.board.state.count(WHITE)
    if gamestate.player == BLACK:
        return black_stones - white_stones
    return white_stones - black_stones


class MCTSNode(object):
    def __init__(self,gamestate, parent=None,move=None):
        self.gamestate=gamestate
        self.parent = parent
        self.move = move
        self.win_counts = {
            BLACK : 0,
            WHITE : 0,
        }
        self.num_rollouts = 0
        self.children = []
        self.unvisited_moves = gamestate.legal_moves()

    def add_random_child(self):
        index = random.randint(0, len(self.unvisited_moves)-1)
        new_move = self.unvisited_moves.pop(index)
        new_gamestate = self.gamestate.apply_move(new_move)
        new_node = MCTSNode(new_gamestate, self, new_move)
        self.children.append(new_node)
        return new_node

    def record_win(self, winner):
        self.win_counts[winner] += 1
        self.num_rollouts += 1

    def can_add_child(self):
        return len(self.unvisited_moves) > 0

    def is_terminal(self):
        return self.gamestate.is_over()

    def winning_frac(self, color):
        return float(self.win_counts[color]) / float(self.num_rollouts)

def uct_score(parent_rollouts, child_rollouts, win_pct, temperature):
    exploration = math.sqrt(math.log(parent_rollouts) / child_rollouts)
    return win_pct + temperature * exploration

class MCTSAgent(Agent):
    def __init__(self, num_rounds, temperature):
        Agent.__init__(self)
        self.num_rounds = num_rounds
        self.temperature = temperature

    def select_move(self,gamestate):
        root = MCTSNode(gamestate)

        for i in range(self.num_rounds):
            node = root
            while (not node.can_add_child()) and (not node.is_terminal()):
                node = self.select_child(node)

            if node.can_add_child():
                node = node.add_random_child()

            winner = self.simulate_random_game(node.gamestate)

            while node is not None:
                node.record_win(winner)
                node = node.parent

        best_move = None
        best_pct = -1.0
        for child in root.children:
            child_pct = child.winning_frac(gamestate.player)
            if child_pct > best_pct:
                best_pct = child_pct
                best_move = child.move
        return best_move

    def select_child(self, node):
        total_rollouts = sum(child.num_rollouts for child in node.children)

        best_score = -1
        best_child = None
        for child in node.children:
            score = uct_score(
                total_rollouts,
                child.num_rollouts,
                child.winning_frac(node.gamestate.player),
                self.temperature)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    @staticmethod
    def simulate_random_game(game):
        R = RandomBot()
        while not game.is_over():
            bot_move = R.select_move(game)
            game = game.apply_move(bot_move)
        return game.winner()
