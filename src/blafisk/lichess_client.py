import requests


class LichessClient:

    def __init__(self, token, api_url="https://lichess.org/api"):
        self.header = {"Authorization": f"Bearer {token}"}
        self.base_url = api_url
        self.session = requests.Session()
        self.session.headers.update(self.header)
        self.other_session = requests.Session()

    def api_get(self, url, params=None, stream=False, timeout=5):
        response = self.session.get(url, params=params, timeout=timeout, stream=stream)
        response.encoding = "utf-8"
        return response

    def api_get_json(self, url, params=None):
        response = self.api_get(url, params=params)
        json_response = response.json()
        return json_response

    def api_get_raw(self, url, params=None):
        response = self.api_get(url, params=params)
        return response.text

    def api_post(
        self,
        url,
        data=None,
        headers=None,
        params=None,
        payload=None,
    ):
        response = self.session.post(
            url, data=data, headers=headers, params=params, json=payload, timeout=5
        )
        json_response = response.json()
        return json_response

    def make_move(self, game_id, move):
        url = f"{self.base_url}/bot/game/{game_id}/move/{move}"
        self.api_post(url)

    def abort(self, game_id: str):
        url = f"{self.base_url}/bot/game/{game_id}/abort"
        self.api_post(url)

    def get_event_stream(self):
        url = f"{self.base_url}/stream/event"
        return self.api_get(url, stream=True, timeout=15)

    def get_game_stream(self, game_id):
        url = f"{self.base_url}/bot/game/stream/{game_id}"
        return self.api_get(url, stream=True, timeout=15)

    def accept_challenge(self, challenge_id):
        url = f"{self.base_url}/challenge/{challenge_id}/accept"
        self.api_post(url)

    def decline_challenge(self, challenge_id):
        url = f"{self.base_url}/challenge/{challenge_id}/decline"
        self.api_post(url)

    def get_ongoing_games(self):
        ongoing_games = []
        url = f"{self.base_url}/account/playing"
        response = self.api_get_json(url)
        ongoing_games = response["nowPlaying"]
        return ongoing_games

    def resign(self, game_id):
        url = f"{self.base_url}/bot/game/{game_id}/resign"
        self.api_post(url)

    def get_game_pgn(self, game_id):
        url = f"{self.base_url}/game/export/{game_id}"
        try:
            return self.api_get_raw(url, params={"pgnInJson": "true"})
        except Exception:
            return ""

    def challenge(self, username, payload):
        return self.api_post("challenge", username, payload=payload)

    def cancel(self, challenge_id):
        self.api_post("cancel", challenge_id)
