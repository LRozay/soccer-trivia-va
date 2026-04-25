"""
soccer_db.py — SQLite schema setup and seed data.
Run this file once to create soccer.db in the project directory.
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "soccer.db")


# ─────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS players (
    player_id   INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    nationality TEXT,
    birth_year  INTEGER,
    position    TEXT
);

CREATE TABLE IF NOT EXISTS clubs (
    club_id     INTEGER PRIMARY KEY,
    club_name   TEXT NOT NULL,
    country     TEXT
);

CREATE TABLE IF NOT EXISTS player_club_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id   INTEGER,
    club_id     INTEGER,
    start_year  INTEGER,
    end_year    INTEGER,
    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (club_id)   REFERENCES clubs(club_id)
);

CREATE TABLE IF NOT EXISTS player_trophies (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id   INTEGER,
    trophy_name TEXT,
    count       INTEGER DEFAULT 1,
    FOREIGN KEY (player_id) REFERENCES players(player_id)
);

CREATE TABLE IF NOT EXISTS player_awards (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id   INTEGER,
    award_name  TEXT,
    year        INTEGER,
    FOREIGN KEY (player_id) REFERENCES players(player_id)
);

CREATE TABLE IF NOT EXISTS club_titles (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    club_id     INTEGER,
    competition TEXT,
    count       INTEGER DEFAULT 1,
    FOREIGN KEY (club_id) REFERENCES clubs(club_id)
);
"""

# ─────────────────────────────────────────────
# Seed Data
# ─────────────────────────────────────────────

PLAYERS = [
    # (player_id, name, nationality, birth_year, position)
    (1,  "Cristiano Ronaldo",   "Portuguese",  1985, "Forward"),
    (2,  "Lionel Messi",        "Argentine",   1987, "Forward"),
    (3,  "Luka Modrić",         "Croatian",    1985, "Midfielder"),
    (4,  "Zinedine Zidane",     "French",      1972, "Midfielder"),
    (5,  "Ronaldinho",          "Brazilian",   1980, "Forward"),
    (6,  "Thierry Henry",       "French",      1977, "Forward"),
    (7,  "Ronaldo Nazário",     "Brazilian",   1976, "Forward"),
    (8,  "Xavi Hernández",      "Spanish",     1980, "Midfielder"),
    (9,  "Andrés Iniesta",      "Spanish",     1984, "Midfielder"),
    (10, "Neymar",              "Brazilian",   1992, "Forward"),
    (11, "Kylian Mbappé",       "French",      1998, "Forward"),
    (12, "Erling Haaland",      "Norwegian",   2000, "Forward"),
    (13, "Robert Lewandowski",  "Polish",      1988, "Forward"),
    (14, "Karim Benzema",       "French",      1987, "Forward"),
    (15, "Sadio Mané",          "Senegalese",  1992, "Forward"),
    (16, "Mohamed Salah",       "Egyptian",    1992, "Forward"),
    (17, "Kevin De Bruyne",     "Belgian",     1991, "Midfielder"),
    (18, "Didier Drogba",       "Ivorian",     1978, "Forward"),
    (19, "Zlatan Ibrahimović",  "Swedish",     1981, "Forward"),
    (20, "Gianluigi Buffon",    "Italian",     1978, "Goalkeeper"),
    (21, "Iker Casillas",       "Spanish",     1981, "Goalkeeper"),
    (22, "Paolo Maldini",       "Italian",     1968, "Defender"),
    (23, "Sergio Ramos",        "Spanish",     1986, "Defender"),
    (24, "Pelé",                "Brazilian",   1940, "Forward"),
    (25, "Diego Maradona",      "Argentine",   1960, "Forward"),
]

CLUBS = [
    (1,  "Real Madrid",       "Spain"),
    (2,  "FC Barcelona",      "Spain"),
    (3,  "Manchester United", "England"),
    (4,  "Manchester City",   "England"),
    (5,  "Liverpool",         "England"),
    (6,  "Arsenal",           "England"),
    (7,  "Chelsea",           "England"),
    (8,  "Juventus",          "Italy"),
    (9,  "AC Milan",          "Italy"),
    (10, "Inter Milan",       "Italy"),
    (11, "PSG",               "France"),
    (12, "Bayern Munich",     "Germany"),
    (13, "Tottenham Hotspur", "England"),
    (14, "Napoli",            "Italy"),
    (15, "Ajax",              "Netherlands"),
    (16, "Atlético Madrid",   "Spain"),
    (17, "Santos",            "Brazil"),
    (18, "Al Nassr",          "Saudi Arabia"),
    (19, "Inter Miami",       "USA"),
    (20, "Dinamo Zagreb",     "Croatia"),
]

# (player_id, club_id, start_year, end_year)  — end_year=None means current
CLUB_HISTORY = [
    # Ronaldo
    (1, 3,  2003, 2009), (1, 1,  2009, 2018), (1, 8,  2018, 2021),
    (1, 3,  2021, 2022), (1, 18, 2023, None),
    # Messi
    (2, 2,  2004, 2021), (2, 11, 2021, 2023), (2, 19, 2023, None),
    # Modrić
    (3, 20, 2003, 2008), (3, 13, 2008, 2012), (3, 1,  2012, None),
    # Zidane
    (4, 8,  1996, 2001), (4, 1,  2001, 2006),
    # Ronaldinho
    (5, 11, 2001, 2003), (5, 2,  2003, 2008), (5, 9,  2008, 2011),
    # Henry
    (6, 6,  1999, 2007), (6, 2,  2007, 2010),
    # R9 Ronaldo
    (7, 2,  1996, 1997), (7, 10, 1997, 2002), (7, 1,  2002, 2007),
    # Xavi
    (8, 2,  1998, 2015),
    # Iniesta
    (9, 2,  2002, 2018),
    # Neymar
    (10, 2,  2013, 2017), (10, 11, 2017, 2023),
    # Mbappé
    (11, 11, 2017, 2024), (11, 1,  2024, None),
    # Haaland
    (12, 4,  2022, None),
    # Lewandowski
    (13, 12, 2014, 2022), (13, 2,  2022, None),
    # Benzema
    (14, 1,  2009, 2023),
    # Mané
    (15, 5,  2016, 2022), (15, 12, 2022, 2023),
    # Salah
    (16, 5,  2017, None),
    # De Bruyne
    (17, 4,  2015, None),
    # Drogba
    (18, 7,  2004, 2012),
    # Ibrahimović
    (19, 15, 2001, 2004), (19, 8, 2004, 2006), (19, 10, 2006, 2009),
    (19, 2,  2009, 2010), (19, 9,  2010, 2012), (19, 11, 2012, 2016),
    (19, 3,  2016, 2017),
    # Buffon
    (20, 8,  2001, 2018),
    # Casillas
    (21, 1,  1999, 2015),
    # Maldini
    (22, 9,  1985, 2009),
    # Ramos
    (23, 16, 2003, 2005), (23, 1, 2005, 2021), (23, 11, 2021, 2023),
    # Pelé
    (24, 17, 1956, 1974),
    # Maradona
    (25, 2,  1982, 1984), (25, 14, 1984, 1991),
]

# (player_id, trophy_name, count)
TROPHIES = [
    # Ronaldo
    (1, "UEFA Champions League", 5),
    (1, "La Liga", 3), (1, "Premier League", 3), (1, "Serie A", 2),
    (1, "UEFA European Championship", 1),

    # Messi
    (2, "UEFA Champions League", 4),
    (2, "La Liga", 10), (2, "Ligue 1", 2),
    (2, "FIFA World Cup", 1),
    (2, "Copa América", 3),

    # Modrić
    (3, "UEFA Champions League", 6),
    (3, "La Liga", 5),

    # Zidane
    (4, "UEFA Champions League", 1),
    (4, "FIFA World Cup", 1),
    (4, "UEFA European Championship", 1),
    (4, "Serie A", 2), (4, "La Liga", 1),

    # Ronaldinho
    (5, "UEFA Champions League", 1),
    (5, "FIFA World Cup", 1),
    (5, "La Liga", 2),
    (5, "Copa América", 1),

    # Henry
    (6, "UEFA Champions League", 1),
    (6, "FIFA World Cup", 1),
    (6, "UEFA European Championship", 1),
    (6, "Premier League", 2),

    # R9 Ronaldo
    (7, "FIFA World Cup", 2),
    (7, "UEFA Champions League", 1),

    # Xavi
    (8, "UEFA Champions League", 4),
    (8, "FIFA World Cup", 1),
    (8, "UEFA European Championship", 2),
    (8, "La Liga", 8),

    # Iniesta
    (9, "UEFA Champions League", 4),
    (9, "FIFA World Cup", 1),
    (9, "UEFA European Championship", 2),
    (9, "La Liga", 9),

    # Neymar
    (10, "UEFA Champions League", 1),
    (10, "La Liga", 3), (10, "Ligue 1", 6),

    # Mbappé
    (11, "FIFA World Cup", 1),
    (11, "Ligue 1", 6),

    # Haaland
    (12, "UEFA Champions League", 1),
    (12, "Premier League", 1),

    # Lewandowski
    (13, "UEFA Champions League", 1),
    (13, "Bundesliga", 8),

    # Benzema
    (14, "UEFA Champions League", 5),
    (14, "La Liga", 5),

    # Mané
    (15, "UEFA Champions League", 1),
    (15, "Premier League", 1),
    (15, "AFCON", 1),

    # Salah
    (16, "UEFA Champions League", 1),
    (16, "Premier League", 1),

    # De Bruyne
    (17, "UEFA Champions League", 1),
    (17, "Premier League", 5),

    # Drogba
    (18, "UEFA Champions League", 1),
    (18, "Premier League", 4),

    # Ibrahimović
    (19, "Serie A", 5),
    (19, "Ligue 1", 4),
    (19, "La Liga", 1),

    # Buffon
    (20, "FIFA World Cup", 1),
    (20, "Serie A", 10),

    # Casillas
    (21, "UEFA Champions League", 3),
    (21, "FIFA World Cup", 1),
    (21, "UEFA European Championship", 2),
    (21, "La Liga", 5),

    # Maldini
    (22, "UEFA Champions League", 5),
    (22, "Serie A", 7),

    # Ramos
    (23, "UEFA Champions League", 4),
    (23, "FIFA World Cup", 1),
    (23, "UEFA European Championship", 2),
    (23, "La Liga", 5),

    # Pelé
    (24, "FIFA World Cup", 3),
    (24, "Copa América", 1),

    # Maradona
    (25, "FIFA World Cup", 1),
    (25, "Serie A", 2),
    (25, "Copa América", 1),
]

# (player_id, award_name, year)
AWARDS = [
    # Ronaldo
    (1, "Ballon d'Or", 2008), (1, "Ballon d'Or", 2013),
    (1, "Ballon d'Or", 2014), (1, "Ballon d'Or", 2016),
    (1, "Ballon d'Or", 2017),
    (1, "FIFA Best Men's Player", 2016), (1, "FIFA Best Men's Player", 2017),

    # Messi
    (2, "Ballon d'Or", 2009), (2, "Ballon d'Or", 2010),
    (2, "Ballon d'Or", 2011), (2, "Ballon d'Or", 2012),
    (2, "Ballon d'Or", 2015), (2, "Ballon d'Or", 2019),
    (2, "Ballon d'Or", 2021), (2, "Ballon d'Or", 2023),
    (2, "FIFA Best Men's Player", 2019), (2, "FIFA Best Men's Player", 2022),

    # Modrić
    (3, "Ballon d'Or", 2018),
    (3, "FIFA Best Men's Player", 2018),

    # Zidane
    (4, "Ballon d'Or", 1998),
    (4, "FIFA World Player of the Year", 1998),
    (4, "FIFA World Player of the Year", 2000),
    (4, "FIFA World Player of the Year", 2003),

    # Ronaldinho
    (5, "Ballon d'Or", 2005),
    (5, "FIFA World Player of the Year", 2004),
    (5, "FIFA World Player of the Year", 2005),

    # R9 Ronaldo
    (7, "Ballon d'Or", 1997),
    (7, "Ballon d'Or", 2002),
    (7, "FIFA World Player of the Year", 1996),
    (7, "FIFA World Player of the Year", 1997),
    (7, "FIFA World Player of the Year", 2002),

    # Benzema
    (14, "Ballon d'Or", 2022),
    (14, "FIFA Best Men's Player", 2022),

    # Lewandowski
    (13, "FIFA Best Men's Player", 2020),
    (13, "FIFA Best Men's Player", 2021),
]

# ─────────────────────────────────────────────
# Setup Function
# ─────────────────────────────────────────────

def setup_database(db_path: str = DB_PATH) -> None:
    """Create and populate the SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.executescript(SCHEMA)

    cursor.executemany(
        "INSERT OR IGNORE INTO players VALUES (?,?,?,?,?)", PLAYERS)
    cursor.executemany(
        "INSERT OR IGNORE INTO clubs VALUES (?,?,?)", CLUBS)
    cursor.executemany(
        "INSERT OR IGNORE INTO player_club_history (player_id, club_id, start_year, end_year) VALUES (?,?,?,?)",
        CLUB_HISTORY)

    # Clear and re-insert trophies/awards to avoid duplication on re-runs
    cursor.execute("DELETE FROM player_trophies")
    cursor.executemany(
        "INSERT INTO player_trophies (player_id, trophy_name, count) VALUES (?,?,?)",
        TROPHIES)

    cursor.execute("DELETE FROM player_awards")
    cursor.executemany(
        "INSERT INTO player_awards (player_id, award_name, year) VALUES (?,?,?)",
        AWARDS)

    conn.commit()
    conn.close()
    print(f"✅ Database ready at: {db_path}")


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # rows as dicts
    return conn


if __name__ == "__main__":
    setup_database()
