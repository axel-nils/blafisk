import os
from dotenv import load_dotenv

from blafisk import LichessBot, LichessClient

load_dotenv()
# BOT_USERNAME = os.getenv("BOT_USERNAME").lower()
LICHESS_TOKEN = os.getenv("LICHESS_TOKEN")


if __name__ == "__main__":
    client = LichessClient(LICHESS_TOKEN)
    games = client.get_ongoing_games()

    if not games:
        print("No ongoing games, checking for challenges...")
        challenge = client.get_challenge()
        if challenge:
            id, username = challenge
            print(f"Challenge from {username} with id {id} accepted.")
            client.accept_challenge(id)
            
    games = client.get_ongoing_games()

    if games:
        print("Playing game!")
        game = games[0]
        bot = LichessBot(client, game)
        bot.run()
    else:
        print("No one to play with :(")
