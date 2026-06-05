import spotipy
from spotipy.oauth2 import SpotifyOAuth

from .config import (
    SPOTIFY_CLIENT_ID,
    SPOTIFY_CLIENT_SECRET,
    SPOTIFY_REDIRECT_URI,
    ACTIVITY_PLAYLISTS,
)

SCOPES = "user-read-playback-state user-modify-playback-state"


class SpotifyController:
    """
    Controls Spotify playback based on the detected activity.

    On first run, spotipy opens a browser for OAuth login and caches the
    token in .cache (gitignored). Subsequent runs use the cached token.
    """

    def __init__(self):
        self._sp = None
        self._current_activity = None

    def connect(self):
        if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
            raise RuntimeError(
                "Spotify credentials missing.\n"
                "Copy .env.example to .env and fill in SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET.\n"
                "Create a Spotify app at https://developer.spotify.com/dashboard"
            )

        auth = SpotifyOAuth(
            client_id=SPOTIFY_CLIENT_ID,
            client_secret=SPOTIFY_CLIENT_SECRET,
            redirect_uri=SPOTIFY_REDIRECT_URI,
            scope=SCOPES,
            open_browser=True,
        )
        self._sp = spotipy.Spotify(auth_manager=auth)
        user = self._sp.me()
        print(f"[Spotify] Connected as: {user['display_name']}")
        self._warn_missing_playlists()
        return self

    def _warn_missing_playlists(self):
        missing = [k for k, v in ACTIVITY_PLAYLISTS.items() if not v]
        if missing:
            print(f"[Spotify] WARNING: no playlist configured for: {missing}")
            print("          Set PLAYLIST_WALKING / PLAYLIST_RUNNING / PLAYLIST_GYM in .env")

    def on_activity_change(self, activity: str):
        """Call this whenever a stable new activity is detected."""
        if activity == self._current_activity or activity == "other":
            return

        playlist_uri = ACTIVITY_PLAYLISTS.get(activity, "")
        if not playlist_uri:
            print(f"[Spotify] No playlist set for '{activity}' — skipping switch")
            self._current_activity = activity
            return

        try:
            playback = self._sp.current_playback()
            if playback and playback.get("is_playing"):
                self._sp.start_playback(context_uri=playlist_uri)
                print(f"[Spotify] Switched to {activity} playlist")
            else:
                print(f"[Spotify] Activity changed to {activity} (Spotify paused — not switching)")
        except spotipy.exceptions.SpotifyException as exc:
            print(f"[Spotify] API error: {exc}")

        self._current_activity = activity
