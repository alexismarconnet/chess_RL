import pytest
import chess
import numpy as np
from environment import ChessEnvironment
from environment.board_encoder import board_to_tensor, get_legal_action_mask, move_to_action, action_to_move

def test_environment_initialization():
    env = ChessEnvironment()
    state = env.reset()
    assert isinstance(state, np.ndarray)
    assert state.shape == (14, 8, 8)
    assert state.dtype == np.float32

def test_legal_mask():
    env = ChessEnvironment()
    mask = env.get_legal_mask()
    assert isinstance(mask, np.ndarray)
    assert mask.shape == (4288,)
    assert mask.dtype == np.float32
    # Au départ, il y a exactement 20 coups légaux aux échecs
    assert np.sum(mask) == 20

def test_board_encoder_tensor_channels():
    env = ChessEnvironment()
    board = env.board
    tensor = board_to_tensor(board)
    assert tensor.shape == (14, 8, 8)
    # Canal du tour de jeu (12) doit être à 1.0 pour les Blancs au début
    assert np.all(tensor[12, :, :] == 1.0)

def test_move_and_action_conversion():
    board = chess.Board()
    for move in board.legal_moves:
        action = move_to_action(move)
        assert isinstance(action, int)
        decoded_move = action_to_move(action, board)
        assert decoded_move == move

def test_environment_step():
    env = ChessEnvironment()
    env.reset()
    legal_moves = env.get_legal_moves()
    assert len(legal_moves) > 0

    move = legal_moves[0]
    next_state, reward, done, info = env.step(move)

    assert isinstance(next_state, np.ndarray)
    assert next_state.shape == (14, 8, 8)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert not done

def test_checkmate_termination():
    env = ChessEnvironment()
    # Mat du berger en 4 coups : 1. f3 e5 2. g4 Qh4#
    env.step(chess.Move.from_uci("f2f3"))
    env.step(chess.Move.from_uci("e7e5"))
    env.step(chess.Move.from_uci("g2g4"))
    next_state, reward, done, info = env.step(chess.Move.from_uci("d8h4"))

    assert done
    assert reward in (1.0, -1.0)
    assert "result" in info
