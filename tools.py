"""
tools.py — The five tool functions the pipeline calls.

Tool A: search_players        — filter the DB by constraints
Tool B: generate_trivia_facts — return facts for a given player (LLM formats the question)
Tool C: check_answer          — verify a user's guess
Tool D: get_hint              — progressive hints for a player
Tool E: web_search            — DuckDuckGo search for current events
"""

import sqlite3
import random
from typing import Optional
from soccer_db import get_connection


# ─────────────────────────────────────────────
# Tool A: Player Search
# ─────────────────────────────────────────────

def search_players(
    won_world_cup:               Optional[bool] = None,
    won_euros:                   Optional[bool] = None,
    won_copa_america:            Optional[bool] = None,
    won_champions_league:        Optional[bool] = None,
    champions_league_min:        Optional[int]  = None,
    won_ballon_dor:              Optional[bool] = None,
    ballon_dor_min:              Optional[int]  = None,
    nationality:                 Optional[str]  = None,
    position:                    Optional[str]  = None,
    played_for_club:             Optional[str]  = None,
    birth_year_min:              Optional[int]  = None,
    birth_year_max:              Optional[int]  = None,
    limit:                       int = 10,
) -> list[dict]:
    """
    Filter players by one or more constraints.
    Returns a list of player dicts with their key stats attached.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Start with all players
    query = "SELECT DISTINCT p.player_id, p.name, p.nationality, p.birth_year, p.position FROM players p"
    params = []

    # Club join if needed
    if played_for_club:
        query += """
            JOIN player_club_history pch ON p.player_id = pch.player_id
            JOIN clubs c ON pch.club_id = c.club_id
        """

    query += " WHERE 1=1"

    if nationality:
        query += " AND LOWER(p.nationality) = LOWER(?)"
        params.append(nationality)

    if position:
        query += " AND LOWER(p.position) = LOWER(?)"
        params.append(position)

    if birth_year_min:
        query += " AND p.birth_year >= ?"
        params.append(birth_year_min)

    if birth_year_max:
        query += " AND p.birth_year <= ?"
        params.append(birth_year_max)

    if played_for_club:
        query += " AND LOWER(c.club_name) LIKE LOWER(?)"
        params.append(f"%{played_for_club}%")

    cursor.execute(query, params)
    players = [dict(row) for row in cursor.fetchall()]

    # Post-filter using trophy/award tables
    def get_trophy_count(player_id, trophy):
        cursor.execute(
            "SELECT count FROM player_trophies WHERE player_id=? AND LOWER(trophy_name) LIKE LOWER(?)",
            (player_id, f"%{trophy}%")
        )
        row = cursor.fetchone()
        return row["count"] if row else 0

    def get_award_count(player_id, award):
        cursor.execute(
            "SELECT COUNT(*) as cnt FROM player_awards WHERE player_id=? AND LOWER(award_name) LIKE LOWER(?)",
            (player_id, f"%{award}%")
        )
        return cursor.fetchone()["cnt"]

    filtered = []
    for p in players:
        pid = p["player_id"]

        if won_world_cup is not None:
            has_wc = get_trophy_count(pid, "FIFA World Cup") > 0
            if has_wc != won_world_cup:
                continue

        if won_euros is not None:
            has_euros = get_trophy_count(pid, "European Championship") > 0
            if has_euros != won_euros:
                continue

        if won_copa_america is not None:
            has_ca = get_trophy_count(pid, "Copa América") > 0
            if has_ca != won_copa_america:
                continue

        if won_champions_league is not None:
            ucl = get_trophy_count(pid, "Champions League")
            has_ucl = ucl > 0
            if has_ucl != won_champions_league:
                continue

        if champions_league_min is not None:
            ucl = get_trophy_count(pid, "Champions League")
            if ucl < champions_league_min:
                continue

        if won_ballon_dor is not None:
            has_bd = get_award_count(pid, "Ballon d'Or") > 0
            if has_bd != won_ballon_dor:
                continue

        if ballon_dor_min is not None:
            bd = get_award_count(pid, "Ballon d'Or")
            if bd < ballon_dor_min:
                continue

        filtered.append(pid)

    if not filtered:
        conn.close()
        return []

    # Enrich matching players with full stats
    result = []
    for pid in filtered[:limit]:
        player = next(p for p in players if p["player_id"] == pid)

        # Trophies
        cursor.execute(
            "SELECT trophy_name, count FROM player_trophies WHERE player_id=? ORDER BY count DESC",
            (pid,)
        )
        player["trophies"] = [dict(r) for r in cursor.fetchall()]

        # Awards
        cursor.execute(
            "SELECT award_name, year FROM player_awards WHERE player_id=? ORDER BY award_name, year",
            (pid,)
        )
        player["awards"] = [dict(r) for r in cursor.fetchall()]

        # Clubs
        cursor.execute("""
            SELECT c.club_name, pch.start_year, pch.end_year
            FROM player_club_history pch
            JOIN clubs c ON pch.club_id = c.club_id
            WHERE pch.player_id = ?
            ORDER BY pch.start_year
        """, (pid,))
        player["clubs"] = [dict(r) for r in cursor.fetchall()]

        result.append(player)

    conn.close()
    return result


# ─────────────────────────────────────────────
# Tool B: Trivia Facts (pipeline generates the question)
# ─────────────────────────────────────────────

def get_player_facts(player_id: int) -> dict | None:
    """
    Return all facts for a single player — the LLM uses these to
    generate a well-grounded trivia question or riddle.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM players WHERE player_id=?", (player_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return None

    player = dict(row)

    cursor.execute(
        "SELECT trophy_name, count FROM player_trophies WHERE player_id=? ORDER BY count DESC",
        (player_id,)
    )
    player["trophies"] = [dict(r) for r in cursor.fetchall()]

    cursor.execute(
        "SELECT award_name, year FROM player_awards WHERE player_id=? ORDER BY award_name, year",
        (player_id,)
    )
    player["awards"] = [dict(r) for r in cursor.fetchall()]

    cursor.execute("""
        SELECT c.club_name, pch.start_year, pch.end_year
        FROM player_club_history pch
        JOIN clubs c ON pch.club_id = c.club_id
        WHERE pch.player_id = ?
        ORDER BY pch.start_year
    """, (player_id,))
    player["clubs"] = [dict(r) for r in cursor.fetchall()]

    conn.close()
    return player


def pick_random_player(position: str = None, nationality: str = None) -> dict | None:
    """Pick a random player from the DB (optionally filtered)."""
    conn = get_connection()
    cursor = conn.cursor()

    query = "SELECT player_id FROM players WHERE 1=1"
    params = []
    if position:
        query += " AND LOWER(position)=LOWER(?)"
        params.append(position)
    if nationality:
        query += " AND LOWER(nationality)=LOWER(?)"
        params.append(nationality)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return None
    pid = random.choice(rows)["player_id"]
    return get_player_facts(pid)


# ─────────────────────────────────────────────
# Tool C: Answer Checker
# ─────────────────────────────────────────────

# Common nickname/alternate spelling map
PLAYER_ALIASES = {
    "mo salah":          "Mohamed Salah",
    "mohammed salah":    "Mohamed Salah",
    "mohammad salah":    "Mohamed Salah",
    "r9":                "Ronaldo Nazário",
    "ronaldo r9":        "Ronaldo Nazário",
    "the phenomenon":    "Ronaldo Nazário",
    "cr7":               "Cristiano Ronaldo",
    "leo messi":         "Lionel Messi",
    "dinho":             "Ronaldinho",
    "van dijk":          "Virgil van Dijk",
    "vvd":               "Virgil van Dijk",
    "taa":               "Trent Alexander-Arnold",
    "trent":             "Trent Alexander-Arnold",
    "modric":            "Luka Modrić",
    "luka modric":       "Luka Modrić",
    "ibra":              "Zlatan Ibrahimović",
    "zlatan":            "Zlatan Ibrahimović",
    "kdb":               "Kevin De Bruyne",
    "lewandowski":       "Robert Lewandowski",
    "lewy":              "Robert Lewandowski",
    "haaland":           "Erling Haaland",
    "mbappe":            "Kylian Mbappé",
    "kylian mbappe":     "Kylian Mbappé",
    "drogba":            "Didier Drogba",
    "henry":             "Thierry Henry",
    "zidane":            "Zinedine Zidane",
    "zizou":             "Zinedine Zidane",
    "pele":              "Pelé",
    "maradona":          "Diego Maradona",
    "diego maradona":    "Diego Maradona",
    "buffon":            "Gianluigi Buffon",
    "casillas":          "Iker Casillas",
    "maldini":           "Paolo Maldini",
    "ramos":             "Sergio Ramos",
    "salah":             "Mohamed Salah",
    "mane":              "Sadio Mané",
    "sadio mane":        "Sadio Mané",
    "neymar":            "Neymar",
    "benzema":           "Karim Benzema",
    "xavi":              "Xavi Hernández",
    "iniesta":           "Andrés Iniesta",
}


def _normalize(name: str) -> str:
    """Lowercase, strip accents roughly, remove punctuation."""
    import unicodedata
    name = name.strip().lower()
    # Normalize accented chars (é→e, ć→c, etc.)
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    return name


def check_answer(player_id: int, guess: str) -> dict:
    """
    Checks whether the user's guess matches the target player.
    Supports: full name, last name, common nickname, alternate spellings.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM players WHERE player_id=?", (player_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return {"correct": False, "correct_name": "Unknown", "explanation": "Player not found."}

    correct_name  = row["name"]
    guess_raw     = guess.strip()
    guess_lower   = guess_raw.lower()
    guess_norm    = _normalize(guess_raw)
    correct_norm  = _normalize(correct_name)

    # 1. Alias lookup
    alias_target = PLAYER_ALIASES.get(guess_lower)
    if alias_target and _normalize(alias_target) == correct_norm:
        is_correct = True
    else:
        name_parts = correct_norm.split()
        is_correct = (
            guess_norm == correct_norm                          # full name match
            or guess_norm in name_parts                        # last name only
            or correct_norm in guess_norm                      # full name inside guess
            or any(guess_norm in part for part in name_parts) # partial part match
            or any(part in guess_norm for part in name_parts if len(part) > 3)  # part inside guess
        )

    return {
        "correct": is_correct,
        "correct_name": correct_name,
        "explanation": (
            f"Correct! The answer is {correct_name}."
            if is_correct else f"Not quite."
        ),
    }


# ─────────────────────────────────────────────
# Tool D: Hint Generator
# ─────────────────────────────────────────────

HINT_LEVELS = ["nationality", "position", "era", "club", "trophy_vague"]

def get_hint(player_id: int, hint_number: int) -> dict:
    """
    Returns the nth hint (1-indexed) for a player.
    Hints go from vague to specific.
    """
    facts = get_player_facts(player_id)
    if not facts:
        return {"hint": "No hint available.", "hint_number": hint_number}

    hints = []
    country = NATIONALITY_TO_COUNTRY.get(facts.get("nationality",""), facts.get("nationality",""))

    # Hint 1: country (natural phrasing)
    hints.append(f"This player is from {country}.")

    # Hint 2: position + era
    decade = (facts["birth_year"] // 10) * 10
    hints.append(f"They play as a {facts['position']} and were born in the {decade}s.")

    # Hint 3: most famous club (deduplicated, most recent)
    seen = []
    for c in facts.get("clubs", []):
        if c["club_name"] not in seen:
            seen.append(c["club_name"])
    if seen:
        hints.append(f"One of their most famous clubs is {seen[-1]}.")

    # Hint 4: a trophy they won
    if facts["trophies"]:
        top = facts["trophies"][0]
        hints.append(f"They have won the {top['trophy_name']}.")

    # Hint 5: an award (if any)
    if facts["awards"]:
        hints.append(f"They have won the {facts['awards'][0]['award_name']}.")

    idx = max(0, min(hint_number - 1, len(hints) - 1))
    return {
        "hint": hints[idx],
        "hint_number": hint_number,
        "total_hints_available": len(hints),
    }


# ─────────────────────────────────────────────
# Tool E: Web Search
# ─────────────────────────────────────────────

def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    DuckDuckGo search — no API key needed.
    Returns a list of {title, href, body} dicts.
    Falls back gracefully if the package isn't installed.
    """
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return results
    except ImportError:
        return [{"title": "Search unavailable",
                 "href": "",
                 "body": "Install duckduckgo_search: pip install duckduckgo-search"}]
    except Exception as e:
        return [{"title": "Search error", "href": "", "body": str(e)}]


# ─────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from soccer_db import setup_database
    setup_database()

    print("=== Players who won World Cup + Ballon d'Or ===")
    results = search_players(won_world_cup=True, won_ballon_dor=True)
    for p in results:
        print(f"  {p['name']} ({p['nationality']})")

    print("\n=== Facts: Luka Modrić ===")
    import pprint
    pprint.pprint(get_player_facts(3))

    print("\n=== Hint 1 for Ronaldo ===")
    print(get_hint(1, 1))

    print("\n=== Check answer ===")
    print(check_answer(1, "ronaldo"))
    print(check_answer(1, "Messi"))


# ─────────────────────────────────────────────
# Tool F: Template-based Question Builder
# ─────────────────────────────────────────────

# Easy templates — single obvious fact
EASY_TEMPLATES = [
    ("Which {nationality} {position} won the FIFA World Cup representing {country} and played for {club}?",
     ["nationality", "position", "world_cup_country", "club"]),
    ("Which {position} has won the Ballon d'Or {ballon_dor_count} time(s) and plays as a {position}?",
     ["position", "ballon_dor_count"]),
    ("Which {nationality} player has won the UEFA Champions League {ucl_count} time(s)?",
     ["nationality", "ucl_count"]),
    ("This {position} was born in the {decade}s, represents {country} internationally, and played for {club}. Who is it?",
     ["nationality", "position", "decade", "club"]),
]

# Medium templates — two constraints
MEDIUM_TEMPLATES = [
    ("I am {nationality} and play as a {position}. I have won the {trophy1} and the {trophy2}. Who am I?",
     ["nationality", "position", "trophy1", "trophy2"]),
    ("Which player won the Champions League {ucl_count} time(s) but never won the FIFA World Cup, "
     "and played for {club}?",
     ["ucl_count", "club"]),
    ("I have won the Ballon d'Or {ballon_dor_count} time(s). I am {nationality} and I am a {position}. Who am I?",
     ["ballon_dor_count", "nationality", "position"]),
    ("This player represents {country} internationally, played for {club1} and {club2}, "
     "and won the {trophy1}. Who are they?",
     ["nationality", "club1", "club2", "trophy1"]),
]

# Hard templates — three+ constraints, one notable absence
HARD_TEMPLATES = [
    ("I have won the Champions League {ucl_count} times. I have won the {award}. "
     "I have never won the FIFA World Cup. I am {nationality}. Who am I?",
     ["ucl_count", "award", "nationality"]),
    ("I am a {position} born in the {decade}s. I won the {trophy1} {t1_count} time(s) "
     "but never won the {missing_trophy}. I played for {club}. Who am I?",
     ["position", "decade", "trophy1", "t1_count", "missing_trophy", "club"]),
    ("I have won the Ballon d'Or {ballon_dor_count} times. I also won the {trophy1} "
     "and the {trophy2}. I never won {missing_trophy}. Who am I?",
     ["ballon_dor_count", "trophy1", "trophy2", "missing_trophy"]),
    ("Which {nationality} {position} has won the Champions League {ucl_count} times, "
     "the {award}, and played for both {club1} and {club2}?",
     ["nationality", "position", "ucl_count", "award", "club1", "club2"]),
]

TEMPLATES_BY_DIFFICULTY = {
    "easy":   EASY_TEMPLATES,
    "medium": MEDIUM_TEMPLATES,
    "hard":   HARD_TEMPLATES,
}

# Trophies that are interesting to mention as absent
NOTABLE_TROPHIES = ["FIFA World Cup", "UEFA Champions League",
                    "UEFA European Championship", "Copa América"]



# Map nationality adjectives to country names for natural phrasing
NATIONALITY_TO_COUNTRY = {
    "Portuguese": "Portugal",   "Argentine":   "Argentina",
    "Croatian":   "Croatia",    "French":      "France",
    "Brazilian":  "Brazil",     "Spanish":     "Spain",
    "Norwegian":  "Norway",     "Polish":      "Poland",
    "Senegalese": "Senegal",    "Egyptian":    "Egypt",
    "Belgian":    "Belgium",    "Ivorian":     "Ivory Coast",
    "Swedish":    "Sweden",     "Italian":     "Italy",
    "German":     "Germany",
}

def build_trivia_question(player: dict, difficulty: str = "medium") -> dict:
    """
    Build a verified trivia question directly from DB facts.
    Returns:
        {
          "question": str,       — the question text
          "answer": str,         — correct player name
          "player_id": int,
          "difficulty": str,
          "facts_used": list,    — which facts appear in the question
        }

    No LLM involved — every claim in the question is DB-verified.
    """
    difficulty = difficulty if difficulty in TEMPLATES_BY_DIFFICULTY else "medium"

    # ── Extract key facts ──────────────────────────────────────
    name        = player["name"]
    nationality = player.get("nationality", "Unknown")
    country     = NATIONALITY_TO_COUNTRY.get(nationality, nationality)
    position    = player.get("position", "player")
    birth_year  = player.get("birth_year") or 1980
    decade      = f"{(birth_year // 10) * 10}"

    trophies    = {t["trophy_name"]: t["count"] for t in player.get("trophies", [])}
    awards      = [a["award_name"] for a in player.get("awards", [])]

    # Deduplicate clubs while preserving order
    seen = []
    for c in player.get("clubs", []):
        if c["club_name"] not in seen:
            seen.append(c["club_name"])
    clubs = seen

    ucl_count   = trophies.get("UEFA Champions League", 0)
    wc_won      = trophies.get("FIFA World Cup", 0) > 0
    euros_won   = trophies.get("UEFA European Championship", 0) > 0
    copa_won    = trophies.get("Copa América", 0) > 0

    ballon_dor_count = sum(1 for a in player.get("awards", []) if "Ballon" in a["award_name"])
    top_award    = awards[0] if awards else None

    # Pick clubs — ensure club1 != club2
    club  = clubs[0] if clubs else "a major club"
    club1 = clubs[0] if clubs else "a major club"
    club2 = clubs[1] if len(clubs) >= 2 else None

    # Notable trophy they DON'T have
    missing_trophy = next(
        (t for t in NOTABLE_TROPHIES if trophies.get(t, 0) == 0), "FIFA World Cup")

    # Non-UCL trophies they DO have (deduplicated)
    other_trophies = [t for t in trophies if "Champions" not in t and trophies[t] > 0]
    trophy1  = other_trophies[0] if other_trophies else (list(trophies.keys())[0] if trophies else "a major trophy")
    trophy2  = next((t for t in other_trophies if t != trophy1), None)
    t1_count = trophies.get(trophy1, 1)

    # ── Fill template slots ────────────────────────────────────
    slots = {
        "nationality":       nationality,
        "country":          country,
        "position":          position,
        "decade":            decade,
        "club":              club,
        "club1":             club1,
        "club2":             club2,
        "ucl_count":         ucl_count,
        "ballon_dor_count":  ballon_dor_count,
        "award":             top_award or "FIFA Best Player",
        "trophy1":           trophy1,
        "trophy2":           trophy2 or trophy1,
        "t1_count":          t1_count,
        "missing_trophy":    missing_trophy,
        "world_cup_country": country,
    }

    # ── Try templates until one is satisfiable ─────────────────
    templates = TEMPLATES_BY_DIFFICULTY[difficulty][:]
    random.shuffle(templates)

    question_text = None
    facts_used    = []

    for template_str, required_fields in templates:
        # Check all required fields have meaningful values
        ok = True
        for field in required_fields:
            val = slots.get(field)
            if val is None or val == 0 or val == "":
                ok = False
                break
            # Don't use UCL count if player has 0 UCL wins
            if field == "club2" and not club2:
                ok = False
                break
            if field == "trophy2" and not trophy2:
                ok = False
                break
            # Don't use World Cup template if player never won it
            if field == "world_cup_country" and not wc_won:
                ok = False
                break
            # Don't use goalkeeper template for non-goalkeepers
            if field == "goalkeeper" and position != "Goalkeeper":
                ok = False
                break
            if field == "ucl_count" and ucl_count == 0:
                ok = False
                break
            # Don't use ballon_dor if player has 0
            if field == "ballon_dor_count" and ballon_dor_count == 0:
                ok = False
                break
            # Don't use award if player has none
            if field == "award" and not top_award:
                ok = False
                break

        if ok:
            try:
                question_text = template_str.format(**slots)
                facts_used    = required_fields
                break
            except KeyError:
                continue

    # ── Fallback: simple guaranteed question ──────────────────
    if not question_text:
        question_text = (
            f"Which {nationality} {position} from the {decade}s "
            f"played for {club}?"
        )
        facts_used = ["nationality", "position", "decade", "club"]

    # ── Uniqueness check ──────────────────────────────────────
    # Build constraint dict from facts_used to verify only 1 player matches
    uniqueness_constraints = {}
    if "ucl_count" in facts_used and ucl_count > 0:
        uniqueness_constraints["champions_league_min"] = ucl_count
    if "nationality" in facts_used:
        uniqueness_constraints["nationality"] = nationality
    if "won_world_cup" in facts_used or "missing_trophy" in facts_used:
        if "FIFA World Cup" == missing_trophy:
            uniqueness_constraints["won_world_cup"] = False
    if "ballon_dor_count" in facts_used and ballon_dor_count > 0:
        uniqueness_constraints["ballon_dor_min"] = ballon_dor_count
    if "position" in facts_used:
        uniqueness_constraints["position"] = position

    if uniqueness_constraints:
        matches = search_players(**uniqueness_constraints, limit=10)
        if len(matches) > 1:
            # Too ambiguous — fall back to a more specific question
            if ucl_count > 0 and ballon_dor_count > 0:
                question_text = (
                    f"Which {nationality} {position} has won the Champions League "
                    f"{ucl_count} time(s) and the Ballon d'Or {ballon_dor_count} time(s)?"
                )
            elif clubs and len(clubs) >= 2:
                question_text = (
                    f"Which {nationality} {position} played for both "
                    f"{clubs[0]} and {clubs[1]}?"
                )
            else:
                question_text = (
                    f"Which {nationality} {position} born in the {decade}s "
                    f"played for {club} and won the {trophy1}?"
                )
            facts_used = ["nationality", "position", "decade", "club", "trophy1"]

    return {
        "question":   question_text,
        "answer":     name,
        "player_id":  player["player_id"],
        "difficulty": difficulty,
        "facts_used": facts_used,
    }


def build_quiz(players: list[dict], difficulty: str = "medium") -> list[dict]:
    """Build a list of verified trivia questions from a list of players."""
    return [build_trivia_question(p, difficulty) for p in players]
