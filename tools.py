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

def check_answer(player_id: int, guess: str) -> dict:
    """
    Checks whether the user's guess matches the target player.
    Returns: {correct: bool, correct_name: str, explanation: str}
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM players WHERE player_id=?", (player_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return {"correct": False, "correct_name": "Unknown", "explanation": "Player not found in database."}

    correct_name = row["name"]
    guess_clean = guess.strip().lower()

    # Flexible matching: full name, last name, common nickname
    name_parts = correct_name.lower().split()
    is_correct = (
        guess_clean == correct_name.lower()
        or guess_clean in name_parts
        or any(guess_clean in part for part in name_parts)
        or correct_name.lower() in guess_clean
    )

    return {
        "correct": is_correct,
        "correct_name": correct_name,
        "explanation": (
            f"✅ Correct! The answer is {correct_name}."
            if is_correct
            else f"❌ Not quite. The answer is {correct_name}."
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

    # Hint 1: nationality
    hints.append(f"I am {facts['nationality']}.")

    # Hint 2: position + era
    decade = (facts["birth_year"] // 10) * 10
    hints.append(f"I am a {facts['position']} who was born in the {decade}s.")

    # Hint 3: one famous club
    if facts["clubs"]:
        famous = sorted(facts["clubs"], key=lambda c: -(c["end_year"] or 9999) + (c["start_year"] or 0))
        hints.append(f"One of my most famous clubs was {famous[0]['club_name']}.")

    # Hint 4: trophy hint (vague)
    if facts["trophies"]:
        top = facts["trophies"][0]
        count = top["count"]
        hints.append(
            f"I have won the {top['trophy_name']} "
            + (f"{count} time(s)." if count > 1 else "at least once.")
        )

    # Hint 5: award hint
    if facts["awards"]:
        award = facts["awards"][0]["award_name"]
        hints.append(f"I have won the {award}.")

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
