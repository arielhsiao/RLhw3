from __future__ import print_function

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

import numpy as np

from PIL import Image, ImageDraw, ImageFont

import itertools
import logging
from six import StringIO
import sys

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

class IllegalMove(Exception):
    pass

class test_illegal(Exception):
    pass

def stack(flat, layers=16):
    """Convert an [4, 4] representation into [layers, 4, 4] with one layers for each value."""
    # representation is what each layer represents
    representation = 2 ** (np.arange(layers, dtype=int) + 1)

    # layered is the flat board repeated layers times
    layered = np.repeat(flat[:,:,np.newaxis], layers, axis=-1)

    # Now set the values in the board to 1 or zero depending whether they match representation.
    # Representation is broadcast across a number of axes
    layered = np.where(layered == representation, 1, 0)
    layered = np.transpose(layered, (2,0,1))
    return layered

'''
def monotonicity_score(logM):
    """
    Calculates the monotonicity score for a 4x4 log-board.
    Rewards rows/columns that are either consistently increasing or decreasing.
    
    For each row, it calculates the score for (left < right) and (left > right)
    and takes the one that is 'more true' (higher score).
    It does the same for columns (top < bottom) and (top > bottom).
    
    The total score is the sum of the best score for rows and the best score for columns.
    """
    
    # --- Row Scores ---
    # Score for all rows being Left-to-Right INCREASING
    # We sum the differences only where the condition is met (A[j] < A[j+1])
    score_lr_inc = 0
    # Score for all rows being Left-to-Right DECREASING
    score_lr_dec = 0
    
    for i in range(4): # For each row
        for j in range(3): # For each adjacent pair
            diff = logM[i, j+1] - logM[i, j]
            if diff > 0:       # Increasing
                score_lr_inc += diff
            elif diff < 0:     # Decreasing
                score_lr_dec += abs(diff) # Add positive value

    # --- Column Scores ---
    # Score for all columns being Top-to-Bottom INCREASING
    score_tb_inc = 0
    # Score for all columns being Top-to-Bottom DECREASING
    score_tb_dec = 0
    
    for j in range(4): # For each column
        for i in range(3): # For each adjacent pair
            diff = logM[i+1, j] - logM[i, j]
            if diff > 0:       # Increasing
                score_tb_inc += diff
            elif diff < 0:     # Decreasing
                score_tb_dec += abs(diff) # Add positive value

    # The final score is the sum of the best direction for rows and the best for columns
    # This rewards the board for having a dominant "gradient"
    return max(score_lr_inc, score_lr_dec) + max(score_tb_inc, score_tb_dec)
'''

class My2048Env(gym.Env):
    metadata = {
        "render_modes": ['ansi', 'human', 'rgb_array'],
        "render_fps": 2,
    }

    def __init__(self):
        # Definitions for game. Board must be square.
        self.size = 4
        self.w = self.size
        self.h = self.size
        self.squares = self.size * self.size

        # Maintain own idea of game score, separate from rewards
        self.score = 0

        # Foul counts for illegal moves
        self.foul_count = 0

        # Members for gym implementation
        self.action_space = spaces.Discrete(4)
        # Suppose that the maximum tile is as if you have powers of 2 across the board.
        layers = self.squares
        self.observation_space = spaces.Box(0, 1, (layers, self.w, self.h), dtype=int)
        
        # TODO: Set negative reward (penalty) for illegal moves (optional)
        self.set_illegal_move_reward(-20)
        
        self.set_max_tile(None)

        # Size of square for rendering
        self.grid_size = 70

        # Initialise seed
        self.seed()

        # Reset ready for a game
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_illegal_move_reward(self, reward):
        """Define the reward/penalty for performing an illegal move. Also need
            to update the reward range for this."""
        # Guess that the maximum reward is also 2**squares though you'll probably never get that.
        # (assume that illegal move reward is the lowest value that can be returned
        self.illegal_move_reward = reward
        self.reward_range = (self.illegal_move_reward, float(2**self.squares))

    def set_max_tile(self, max_tile):
        """Define the maximum tile that will end the game (e.g. 2048). None means no limit.
           This does not affect the state returned."""
        assert max_tile is None or isinstance(max_tile, int)
        self.max_tile = max_tile

    # Implement gym interface
    def step(self, action):
        """Perform one step of the game. This involves moving and adding a new tile."""
        logging.debug("Action {}".format(action))
        score = 0
        done = None
        info = {
            'illegal_move': False,
            'highest': 0,
            'score': 0,
        }
        after_matrix = self.Matrix.copy()

        try:
            # assert info['illegal_move'] == False
            pre_matrix = self.Matrix.copy()
            log_pre_state = np.log2(pre_matrix + 1.0)
            score = float(self.move(action))
            self.score += score
            after_matrix = self.Matrix.copy()
            log_after_state = np.log2(after_matrix + 1.0)
            assert score <= 2**(self.w*self.h)

            self.add_tile()
            done = self.isend()
            tan_score = np.tanh(np.log2(score + 1)/3) 
            reward = float(score)

            # TODO: Add reward according to weighted states (optional)
            ''' no try
            weight = np.array([
                    [1  , 0.5  , 0.5  , 1 ],
                    [0.5  , 0  , 0  , 0.5  ],
                    [0.5  , 0  , 0  , 0.5  ],
                    [1  , 0.5  , 0.5  , 1  ]
                    ])
            '''

            # weight =  [[1.   0.95 0.9  0.85]
            # [0.65 0.7  0.75 0.8 ]
            # [0.6  0.55 0.5  0.45]
            # [0.25 0.3  0.35 0.4 ]]

            # Automatically generate snake-like weight matrix
            # base = np.linspace(1.0, 0.2, 16).reshape(4, 4)
            # weight = np.zeros_like(base)
            # snake_indices = [(i, j if i % 2 == 0 else 3 - j) for i in range(4) for j in range(4)]
            # for idx, (x, y) in enumerate(snake_indices):
            #     weight[x, y] = 1.0 - 0.05 * idx  # decrease gradually along snake path
            #print("weight = ",weight)
            # weight = np.array([
            #     [0.072,  0.041,   0.023,   0.012],
            #     [0.14,   0.078,   0.042,   0.022],
            #     [0.27,   0.15,    0.079,   0.041],
            #     [0.52,   0.30,    0.16,    0.083]
            # ])
            # weight = np.array([[0.0625,   0.03125,  0.015625,  0.0078125],
            # [0.125,    0.0625,   0.03125,   0.015625 ],
            # [0.25,     0.125,    0.0625,    0.03125  ],
            # [0.5,      0.25,     0.125,     0.0625   ]])
            weight = np.array([[0.0625,   0.03125,  0.015625,  0.0078125],
            [0.125,    0.0625,   0.03125,   0.015625 ],
            [0.25,     0.125,    0.0625,    0.03125  ],
            [0.5,      0.25,     0.125,     0.0625   ]])
            weighted_score = np.sum((log_after_state - log_pre_state) * weight)
            #reward += 0.02 * weighted_score
            reward += weighted_score

            # curve_prelog = -(np.sum(np.abs(log_pre_state[:,1:]-log_pre_state[:,:-1]))+np.sum(np.abs(log_pre_state[1:, :] - log_pre_state[:-1, :])))
            # curve_afterlog = -(np.sum(np.abs(log_after_state[:,1:]-log_after_state[:,:-1]))+np.sum(np.abs(log_after_state[1:, :] - log_after_state[:-1, :])))
            # curve_delta = curve_afterlog - curve_prelog 
            # reward += 0.05 * curve_delta

            # prev_max = np.max(pre_matrix)
            # curr_max = np.max(after_matrix)
            # if curr_max > prev_max:
            #     step_bonus = 0.1 * np.log2(curr_max)
            #     reward += step_bonus

            def smoothness(logM):
                diff_x = np.sum(np.abs(np.diff(logM, axis=0)))
                diff_y = np.sum(np.abs(np.diff(logM, axis=1)))
                return -(diff_x + diff_y) 
            delta_smooth = smoothness(log_after_state) - smoothness(log_pre_state)
            reward += 0.05*delta_smooth

            num_empty_tiles = np.sum(self.Matrix == 0)   # count how many cells are empty
            reward += 0.0005 * num_empty_tiles

            #print("tan_core, weighted_score, curve_delt =", tan_score, weighted_score, curve_delta, " -> final:", reward)
            #delta_monotonicity = monotonicity_score(log_after_state) - monotonicity_score(log_pre_state)
            #reward += 0.001 * delta_monotonicity
            
        except IllegalMove:
            logging.debug("Illegal move")
            info['illegal_move'] = True
            reward = self.illegal_move_reward + self.illegal_move_reward*0.08*self.foul_count
            self.foul_count += 1
            if self.foul_count >= 40: 
                done = True
            # TODO: Modify this part for the agent to have a chance to explore other actions (optional)
            
        truncate = False
        info['highest'] = self.highest()
        info['score']   = self.score

        # Return observation (board state), reward, done, truncate and info dict
        return stack(self.Matrix), reward, done, truncate, info
    

    def reset(self, seed=None, options=None):
        self.seed(seed=seed)
        self.Matrix = np.zeros((self.h, self.w), int)
        self.score = 0
        self.foul_count = 0

        logging.debug("Adding tiles")
        self.add_tile()
        self.add_tile()

        return stack(self.Matrix), {}

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        s = 'Score: {}\n'.format(self.score)
        s += 'Highest: {}\n'.format(self.highest())
        npa = np.array(self.Matrix)
        grid = npa.reshape((self.size, self.size))
        s += "{}\n".format(grid)
        outfile.write(s)
        return outfile

    # Implement 2048 game
    def add_tile(self):
        """Add a tile, probably a 2 but maybe a 4"""
        possible_tiles = np.array([2, 4])
        tile_probabilities = np.array([0.9, 0.1])
        val = self.np_random.choice(possible_tiles, 1, p=tile_probabilities)[0]
        empties = self.empties()
        assert empties.shape[0]
        empty_idx = self.np_random.choice(empties.shape[0])
        empty = empties[empty_idx]
        logging.debug("Adding %s at %s", val, (empty[0], empty[1]))
        self.set(empty[0], empty[1], val)

    def get(self, x, y):
        """Return the value of one square."""
        return self.Matrix[x, y]

    def set(self, x, y, val):
        """Set the value of one square."""
        self.Matrix[x, y] = val

    def empties(self):
        """Return a 2d numpy array with the location of empty squares."""
        return np.argwhere(self.Matrix == 0)

    def highest(self):
        """Report the highest tile on the board."""
        return np.max(self.Matrix)

    def move(self, direction, trial=False):
        """Perform one move of the game. Shift things to one side then,
        combine. directions 0, 1, 2, 3 are up, right, down, left.
        Returns the score that [would have] got."""
        if not trial:
            if direction == 0:
                logging.debug("Up")
            elif direction == 1:
                logging.debug("Right")
            elif direction == 2:
                logging.debug("Down")
            elif direction == 3:
                logging.debug("Left")

        changed = False
        move_score = 0
        dir_div_two = int(direction / 2)
        dir_mod_two = int(direction % 2)
        shift_direction = dir_mod_two ^ dir_div_two # 0 for towards up left, 1 for towards bottom right

        # Construct a range for extracting row/column into a list
        rx = list(range(self.w))
        ry = list(range(self.h))

        if dir_mod_two == 0:
            # Up or down, split into columns
            for y in range(self.h):
                old = [self.get(x, y) for x in rx]
                (new, ms) = self.shift(old, shift_direction)
                move_score += ms
                if old != new:
                    changed = True
                    if not trial:
                        for x in rx:
                            self.set(x, y, new[x])
        else:
            # Left or right, split into rows
            for x in range(self.w):
                old = [self.get(x, y) for y in ry]
                (new, ms) = self.shift(old, shift_direction)
                move_score += ms
                if old != new:
                    changed = True
                    if not trial:
                        for y in ry:
                            self.set(x, y, new[y])
        if changed != True:
            raise IllegalMove

        return move_score

    def combine(self, shifted_row):
        """Combine same tiles when moving to one side. This function always
           shifts towards the left. Also count the score of combined tiles."""
        move_score = 0
        combined_row = [0] * self.size
        skip = False
        output_index = 0
        for p in pairwise(shifted_row):
            if skip:
                skip = False
                continue
            combined_row[output_index] = p[0]
            if p[0] == p[1]:
                combined_row[output_index] += p[1]
                move_score += p[0] + p[1]
                # Skip the next thing in the list.
                skip = True
            output_index += 1
        if shifted_row and not skip:
            combined_row[output_index] = shifted_row[-1]

        return (combined_row, move_score)

    def shift(self, row, direction):
        """Shift one row left (direction == 0) or right (direction == 1), combining if required."""
        length = len(row)
        assert length == self.size
        assert direction == 0 or direction == 1

        # Shift all non-zero digits up
        shifted_row = [i for i in row if i != 0]

        # Reverse list to handle shifting to the right
        if direction:
            shifted_row.reverse()

        (combined_row, move_score) = self.combine(shifted_row)

        # Reverse list to handle shifting to the right
        if direction:
            combined_row.reverse()

        assert len(combined_row) == self.size
        return (combined_row, move_score)

    def isend(self):
        """Has the game ended. Game ends if there is a tile equal to the limit
           or there are no legal moves. If there are empty spaces then there
           must be legal moves."""

        if self.max_tile is not None and self.highest() == self.max_tile:
            return True

        for direction in range(4):
            try:
                self.move(direction, trial=True)
                # Not the end if we can do any move
                return False
            except IllegalMove:
                pass
        return True

    def get_board(self):
        """Retrieve the whole board, useful for testing."""
        return self.Matrix

    def set_board(self, new_board):
        """Retrieve the whole board, useful for testing."""
        self.Matrix = new_board