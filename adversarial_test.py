import argparse, io, itertools, json, math, operator, os, random, signal, sys, unittest
from unittest.mock import patch

import pacman, time, layout, textDisplay
from game import Agent
from ghostAgents import RandomGhost, DirectionalGhost

import adversarialAgents as adv

SEED = "testing"


def run(layout_name, agent, ghosts, num_games=1, timeout=30):
    layout_obj = layout.getLayout(layout_name, 3)
    display = textDisplay.NullGraphics()
    return pacman.runGames(
        layout_obj,
        agent,
        ghosts,
        display,
        num_games,
        False,
        catchExceptions=False,
        timeout=timeout,
    )


class AgentTests(unittest.TestCase):
    def test_expected_initial_move(self):
        """Agent produces expected initial move"""
        for agent_cls in ("MinimaxAgent", "AlphaBetaAgent","ExpectimaxAgent"):
            random.seed(SEED)
            agent = getattr(adv, agent_cls)(depth=2)
            with patch("sys.stdout", new_callable=io.StringIO) as f:
                games = run(
                    "smallClassic", agent, [DirectionalGhost(i + 1) for i in range(2)]
                )

            self.assertEqual(len(games), 1)
            self.assertFalse(games[0].agentTimeout, f"{agent_cls} agent timed-out preventing testing")

            self.assertGreaterEqual(len(games[0].moveHistory), 1)
            agent_index, move = games[0].moveHistory[0]
            self.assertIn(move, {"East","West"}, f"{agent_cls} agent didn't produce expected initial move")

    def test_expectimax_in_trapped_classic(self):
        """Expectimax wins and loses in trappedClassic layout"""
        random.seed(SEED)
        agent = adv.ExpectimaxAgent(depth=3)
        with patch("sys.stdout", new_callable=io.StringIO) as f:
            games = run(
                "trappedClassic",
                agent,
                [RandomGhost(i + 1) for i in range(2)],
                num_games=20,
            )
        fraction_win = sum(game.state.isWin() for game in games) / len(games)
        self.assertTrue(0 < fraction_win < 1, "We should observe both wins and losses")
        for game in games:
            if game.state.isWin():
                self.assertIn(
                    game.state.getScore(),
                    {531, 532},
                    "Winning score doesn't match 531 or 523",
                )
            else:
                self.assertEqual(
                    game.state.getScore(), -502, "Losing score doesn't match -502"
                )

    def test_alphabeta_pruning_performance(self):
        """Minimax with alpha-beta pruning is faster than without pruning"""
        random.seed(SEED)
        with patch("sys.stdout", new_callable=io.StringIO) as f:
            agent = adv.AlphaBetaAgent(depth=3)
            start_time = time.time()
            games = run(
                "mediumClassic",
                agent,
                [DirectionalGhost(i + 1) for i in range(2)],
                num_games=5,
            )
            alphabeta_time = time.time() - start_time

            agent = adv.MinimaxAgent(depth=3)
            start_time = time.time()
            games = run(
                "mediumClassic",
                agent,
                [DirectionalGhost(i + 1) for i in range(2)],
                num_games=5,
            )
            minimax_time = time.time() - start_time

        self.assertLessEqual(alphabeta_time, minimax_time * 0.75, "Time of alpha-beta pruning should be less than 75% of minimax without pruning")

class EvaluationFunctionTests(unittest.TestCase):
    def test_eval_fn_perf(self):
        """Evaluation function achieves minimum performance specification"""
        random.seed(SEED)
        with patch("sys.stdout", new_callable=io.StringIO) as f:
            params = f"-l smallClassic -p ExpectimaxAgent -a evalFn=better -q -n {20} -c"
            games = pacman.runGames(**pacman.readCommand(params.split(' ')))

        fraction_timeout = sum(game.agentTimeout for game in games) / len(games)
        fraction_win = sum(game.state.isWin() for game in games) / len(games)
        self.assertEqual(fraction_timeout, 0.0, "One or more games timed out")
        self.assertGreaterEqual(fraction_win, 0.75, "Win rate is below 75%")

if __name__ == '__main__':
    unittest.main(argv=sys.argv[:1])
