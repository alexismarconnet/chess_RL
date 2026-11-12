import pytest
import chess
import numpy as np
import torch
from agent import GreedyAgent, EpsilonGreedyAgent, UCBAgent, MCTSAgent
from training import ReplayBuffer, Trainer
from environment import ChessEnvironment

def test_agents_selection():
    board = chess.Board()
    agents = [
        ("Greedy", GreedyAgent()),
        ("Epsilon-Greedy", EpsilonGreedyAgent(epsilon=0.2)),
        ("UCB", UCBAgent(c_param=1.41)),
        ("MCTS (temp=1.0)", MCTSAgent(num_simulations=5, temperature=1.0)),
        ("MCTS (temp=0.0)", MCTSAgent(num_simulations=5, temperature=0.0))
    ]

    for name, agent in agents:
        move = agent.select_action(board)
        assert move in board.legal_moves, f"L'agent {name} a joué un coup illégal : {move}"

def test_mcts_policy_generation():
    agent = MCTSAgent(num_simulations=10)
    board = chess.Board()
    move, policy = agent.get_action_and_policy(board, temperature=1.0, add_noise=True)

    assert move in board.legal_moves
    assert isinstance(policy, np.ndarray)
    assert policy.shape == (4288,)
    assert np.isclose(np.sum(policy), 1.0, atol=1e-5)

def test_replay_buffer_and_trainer():
    env = ChessEnvironment()
    agent = MCTSAgent(num_simulations=5)
    buffer = ReplayBuffer(capacity=100)
    trainer = Trainer(agent, lr=1e-3)

    state = env.reset()
    policy = np.zeros(4288, dtype=np.float32)
    policy[0] = 1.0
    value = 1.0

    buffer.push(state, policy, value)
    buffer.push(state, policy, -1.0)

    assert len(buffer) == 2

    # Test train step
    p_loss, v_loss = trainer.train_step(buffer, batch_size=2)
    assert isinstance(p_loss, float)
    assert isinstance(v_loss, float)
    assert p_loss >= 0.0
    assert v_loss >= 0.0
