import os
from dotenv import load_dotenv

from blafisk import LichessBot, LichessClient

load_dotenv()
# BOT_USERNAME = os.getenv("BOT_USERNAME").lower()
LICHESS_TOKEN = os.getenv("LICHESS_TOKEN")


if __name__ == "__main__":
    client = LichessClient(LICHESS_TOKEN)
    games = client.get_ongoing_games()
    if games:
        game = games[0]
        bot = LichessBot(client, game)
        bot.run()
    else:
        print("No games")
