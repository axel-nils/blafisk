from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import torch
import chess
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """Configuration for move generation.
    
    Args:
        temperature: Controls randomness (0.1=deterministic, 1.0=neutral, 2.0=very random)
        top_p: Nucleus sampling - only consider moves with cumulative probability <= top_p
        top_k: Only consider the top K most likely moves
        num_suggestions: Number of move candidates to return (for suggest_moves)
        max_new_tokens: Maximum moves to generate (for play_game only, not predict_next_move)
    """
    temperature: float = 0.9  # Slightly deterministic for better play
    top_p: Optional[float] = 0.9  # Filter out unlikely moves
    top_k: Optional[int] = None  # Optional additional filtering
    num_suggestions: int = 1
    max_new_tokens: int = 100  # Reasonable max game length


class ChessLLM:
    """Chess move prediction using LLM."""
    
    def __init__(self, model_id: str = "lazy-guy12/chess-llama", device: str = "cuda"):
        """
        Initialize the Chess LLM.
        
        Args:
            model_id: HuggingFace model identifier
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device
        self.model_id = model_id
        
        self.config = AutoConfig.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float16 if device == "cuda" else torch.float32
        )
        self.model = self.model.to(device)
        self.model.eval()
        
        # Cache config values
        self.eos_token_id = self.config.eos_token_id
        self.id_to_move = {v: k for k, v in self.tokenizer.get_vocab().items()}
    
    def _prepare_inputs(self, board: chess.Board, result: str = "0-1") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare model inputs from a chess board position.
        
        Args:
            board: Current chess board state
            result: Game result prefix (e.g., "0-1", "1-0", "1/2-1/2")
            
        Returns:
            Tuple of (input_ids, attention_mask, position_ids)
        """
        # Get move history in UCI format
        move_history = " ".join([move.uci() for move in board.move_stack])
        text = f"{result} {move_history}".strip()
        
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.long, device=self.device).unsqueeze(0)
        
        return input_ids, attention_mask, position_ids
    
    def _apply_sampling(self, logits: torch.Tensor, gen_config: GenerationConfig) -> torch.Tensor:
        """
        Apply temperature and sampling strategies to logits.
        
        Args:
            logits: Raw model logits
            gen_config: Generation configuration
            
        Returns:
            Modified logits
        """
        # Apply temperature
        if gen_config.temperature != 1.0:
            logits = logits / gen_config.temperature
        
        # Apply top-k filtering
        if gen_config.top_k is not None:
            top_k = min(gen_config.top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Apply top-p (nucleus) filtering
        if gen_config.top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > gen_config.top_p
            # Keep at least one token
            sorted_indices_to_remove[..., 0] = False
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        return logits
    
    def predict_next_move(
        self,
        board: chess.Board,
        result: Optional[str] = None,
        gen_config: Optional[GenerationConfig] = None
    ) -> str:
        """
        Predict the next move for the given board position.
        
        Args:
            board: Current chess board state
            result: Game result prefix (e.g., "0-1", "1-0", "1/2-1/2"). 
                   If None, automatically set based on whose turn it is:
                   - White to move: "1-0" (predict move that helps White win)
                   - Black to move: "0-1" (predict move that helps Black win)
            gen_config: Generation configuration (uses defaults if None)
            
        Returns:
            UCI string of the predicted move
        """
        if gen_config is None:
            gen_config = GenerationConfig()
        
        # Auto-determine result based on whose turn it is
        if result is None:
            result = "1-0" if board.turn == chess.WHITE else "0-1"
        
        # Prepare inputs
        input_ids, attention_mask, position_ids = self._prepare_inputs(board, result)
        past_key_values = None
        
        # Run model
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
        
        # Sample output
        logits = outputs.logits[0, -1]
        logits = self._apply_sampling(logits, gen_config)
        
        # Choose best legal move
        sorted_moves = torch.argsort(logits, descending=True).tolist()
        legal_moves = set(x.uci() for x in board.legal_moves)
        
        for token_id in sorted_moves:
            move_uci = self.id_to_move.get(token_id)
            if move_uci in legal_moves:
                return move_uci
        
        # Fallback: return any legal move if no valid move found
        return next(iter(board.legal_moves)).uci()
    
    def suggest_moves(
        self,
        board: chess.Board,
        result: Optional[str] = None,
        gen_config: Optional[GenerationConfig] = None
    ) -> List[Tuple[str, float]]:
        """
        Suggest multiple candidate moves with their scores.
        
        Args:
            board: Current chess board state
            result: Game result prefix. If None, automatically set based on whose turn it is.
            gen_config: Generation configuration (uses defaults if None)
            
        Returns:
            List of (move_uci, score) tuples, sorted by score
        """
        if gen_config is None:
            gen_config = GenerationConfig()
        
        # Auto-determine result based on whose turn it is
        if result is None:
            result = "1-0" if board.turn == chess.WHITE else "0-1"
        
        # Prepare inputs
        input_ids, attention_mask, position_ids = self._prepare_inputs(board, result)
        past_key_values = None
        
        # Run model
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
        
        # Sample output
        logits = outputs.logits[0, -1]
        logits = self._apply_sampling(logits, gen_config)
        probs = torch.softmax(logits, dim=-1)
        
        # Get legal moves with their scores
        legal_moves = set(x.uci() for x in board.legal_moves)
        suggestions = []
        
        for token_id in range(len(probs)):
            move_uci = self.id_to_move.get(token_id)
            if move_uci in legal_moves:
                score = probs[token_id].item()
                suggestions.append((move_uci, score))
        
        # Sort by score and return top N
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:gen_config.num_suggestions]
    
    def play_game(
        self,
        initial_board: Optional[chess.Board] = None,
        initial_moves: Optional[str] = None,
        result: str = "0-1",  # Keep default for backwards compatibility with full game playing
        gen_config: Optional[GenerationConfig] = None,
        verbose: bool = True
    ) -> chess.Board:
        """
        Play a complete game from a starting position.
        
        Args:
            initial_board: Starting board position (uses starting position if None)
            initial_moves: Space-separated UCI moves to initialize the board
            result: Game result prefix (determines which side the model plays for)
            gen_config: Generation configuration
            verbose: Whether to print moves as they're generated
            
        Returns:
            Final board state
        """
        if gen_config is None:
            gen_config = GenerationConfig()
        
        # Initialize board
        board = initial_board if initial_board is not None else chess.Board()
        if initial_moves:
            for move in initial_moves.split():
                board.push_uci(move)
        
        # Prepare inputs
        input_ids, attention_mask, position_ids = self._prepare_inputs(board, result)
        past_key_values = None
        
        # Generation loop
        for _ in range(gen_config.max_new_tokens):
            # Run model
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True
                )
            
            logits = outputs.logits[0, -1]
            past_key_values = outputs.past_key_values
            
            # Apply sampling
            logits = self._apply_sampling(logits, gen_config)
            
            # Choose best legal move
            sorted_moves = torch.argsort(logits, descending=True).tolist()
            legal_moves = set(x.uci() for x in board.legal_moves)
            
            next_id = None
            next_uci = None
            for token_id in sorted_moves:
                move_uci = self.id_to_move.get(token_id)
                if move_uci in legal_moves:
                    next_id = token_id
                    next_uci = move_uci
                    break
            
            if next_uci is None:
                break
            
            board.push_uci(next_uci)
            
            # Update inputs for next iteration
            input_ids = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
            attention_mask = torch.cat([attention_mask, torch.ones_like(input_ids, dtype=torch.long)], dim=-1)
            position_ids = position_ids[:, -1:] + 1
            
            if verbose:
                print(next_uci)
            
            if next_id == self.eos_token_id or board.is_game_over():
                break
        
        return board


def main():
    """Example usage of the ChessLLM class."""
    chess_llm = ChessLLM(device="cuda")
    
    print("Example 1: Predict single move (auto-detects whose turn it is)")
    board = chess.Board()
    initial_moves = "e2e4 e7e5 g1f3 g8f6 f3e5 b8c6 e5c6 d7c6 b1c3 f8c5 f1c4"
    for move in initial_moves.split():
        board.push_uci(move)
    
    print(f"Current turn: {'White' if board.turn == chess.WHITE else 'Black'}")
    next_move = chess_llm.predict_next_move(board)
    print(f"Next move: {next_move}\n")
    
    print("Example 2: Get top 5 move suggestions")
    gen_config = GenerationConfig(num_suggestions=5)
    suggestions = chess_llm.suggest_moves(board, gen_config=gen_config)
    for i, (move, score) in enumerate(suggestions, 1):
        print(f"{i}. {move}: {score:.4f}")
    print()
    
    print("Example 3: Play complete game (Black to move)")
    board = chess.Board()
    for move in initial_moves.split():
        board.push_uci(move)
    
    final_board = chess_llm.play_game(
        initial_board=board.copy(),
        result="0-1",  # Play as Black (since it's Black's turn after the initial moves)
        verbose=True
    )
    print(f"\nFinal position:\n{final_board}")


if __name__ == "__main__":
    main()