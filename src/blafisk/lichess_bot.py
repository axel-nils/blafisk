import random
import json
import chess
from .lichess_client import LichessClient
from .chess_llm import ChessLLM


class LichessBot:
    def __init__(self, client, game):
        self.client: LichessClient = client
        self.llm = ChessLLM()
        self.game_id = game["gameId"]
        self.is_white = game["color"] == "white"
        self.is_my_turn = game["isMyTurn"]
        self.board = None
        self.last_move_count = None

    def stream_moves(self):
        """Yields UCI moves from the game stream."""
        response = self.client.get_game_stream(self.game_id)
        buffer = ""

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                if buffer:
                    payload = self.parse_payload_from_buffer(buffer)
                    buffer = ""
                    if not payload:
                        continue

                    if payload.get("type") == "gameFull":
                        yield from self.handle_game_full(payload)
                        continue

                    if payload.get("type") == "gameState":
                        yield from self.handle_game_state(payload)
                continue

            buffer += line + "\n"

    def parse_payload_from_buffer(self, buffer):
        data = None
        for entry in buffer.strip().splitlines():
            if entry.startswith("data:"):
                data = entry.split(":", 1)[1].strip()
            elif entry.startswith("{") and entry.endswith("}"):
                data = entry.strip()
        if not data:
            return None
        return json.loads(data)

    def handle_game_full(self, payload):
        """Initialize board and possibly play first move if it's our turn"""
        self.board = chess.Board()
        moves_str = payload.get("state", {}).get("moves", "")
        played = moves_str.split()
        self.last_move_count = len(played)

        for move in played:
            self.board.push_uci(move)

        if self.board.turn == self.is_white:
            my_move = self._select_move()
            if my_move:
                self.client.make_move(self.game_id, my_move)
                yield {"move": my_move, "by_us": True}
            self.last_move_count += 1

    def handle_game_state(self, payload):
        """Yield new moves and play if it's our turn"""
        moves = payload.get("moves", "").split()
        old_count = self.last_move_count
        new_moves = moves[old_count:]

        for i, move in enumerate(new_moves):
            global_index = old_count + i
            by_us = (global_index % 2 == 0 and self.is_white) or (
                global_index % 2 == 1 and not self.is_white
            )
            yield {"move": move, "by_us": by_us}

        self.last_move_count = len(moves)

        if self.board.turn == self.is_white:
            my_move = self._select_move()
            if my_move:
                self.client.make_move(self.game_id, my_move)
                yield {"move": my_move, "by_us": True}
            self.last_move_count += 1

    def _select_move(self):
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return None
        # move = random.choice(legal_moves).uci()
        move = self.llm.predict_next_move(self.board)
        return move

    def run(self):
        for item in self.stream_moves():
            move = item["move"]
            by_us = item["by_us"]

            print(f"{'We' if by_us else 'Opponent'} played: {move}")
            self.board.push_uci(move)
