import json
from pathlib import Path
from flask import Flask, render_template, abort

app = Flask(__name__, static_folder="style", static_url_path="/static")

BASE_DIR = Path(__file__).parent.parent
RANKED_LIST_PATH = BASE_DIR / "data" / "output" / "ranked_list.json"
SCORED_DIR = BASE_DIR / "data" / "output" / "scored"


def load_ranked_list() -> dict:
    if not RANKED_LIST_PATH.exists():
        return {}
    return json.loads(RANKED_LIST_PATH.read_text(encoding="utf-8"))


def load_scorecard(candidate_id: str) -> dict:
    path = SCORED_DIR / f"{candidate_id}_scorecard.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@app.route("/")
def index():
    ranked_list = load_ranked_list()
    if not ranked_list:
        return "<h2>No results yet. Run the pipeline first.</h2>", 404
    return render_template("index.html", data=ranked_list)


@app.route("/candidate/<candidate_id>")
def scorecard(candidate_id: str):
    sc = load_scorecard(candidate_id)
    if not sc:
        abort(404)
    return render_template("scorecard.html", scorecard=sc, candidate_id=candidate_id)


if __name__ == "__main__":
    from config import DASHBOARD_HOST, DASHBOARD_PORT, DASHBOARD_DEBUG
    app.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=DASHBOARD_DEBUG)