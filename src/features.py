import re
import pandas as pd
import numpy as np

TARGET_LABELS = {
    0: "Home win",
    1: "Draw",
    2: "Away win",
}


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def parse_match_date(s):
    """Parse both normal date strings and the old FIFA CSV numeric day offset.

    The old matches.csv stores match_date as days from 1970-01-01, e.g. -14417 = 1930-07-13.
    Plain pd.to_datetime(-14417) would produce 1969-12-31 because pandas treats it as nanoseconds.
    """
    if isinstance(s, pd.Series):
        raw = s.copy()
        num = pd.to_numeric(raw, errors="coerce")
        non_null = raw.notna().sum()
        numeric_share = float(num.notna().sum()) / max(1, int(non_null))
        if numeric_share > 0.8:
            return pd.to_datetime(num, unit="D", origin="1970-01-01", errors="coerce")
        return pd.to_datetime(raw, errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def _safe_div(a, b, default=0.0):
    try:
        if b is None or float(b) == 0 or pd.isna(b):
            return default
        return float(a) / float(b)
    except Exception:
        return default


def _stage_knockout_flag(stage):
    s = str(stage).lower()
    return 0 if ("group" in s or "pool" in s) else 1


# -----------------------------------------------------------------------------
# Old three-table dataset: matches.csv + team_appearances.csv + teams_rows.csv
# -----------------------------------------------------------------------------

def prepare_matches(matches):
    df = matches.copy()
    df["match_date"] = parse_match_date(df["match_date"])
    for c in ["home_team_score", "away_team_score"]:
        df[c] = to_num(df[c])
    df = df.dropna(subset=["match_date", "home_team_score", "away_team_score"])

    conditions = [
        df["home_team_score"] > df["away_team_score"],
        df["home_team_score"] == df["away_team_score"],
        df["home_team_score"] < df["away_team_score"],
    ]
    df["target"] = np.select(conditions, [0, 1, 2], default=np.nan)
    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype(int)
    df["target_name"] = df["target"].map(TARGET_LABELS)

    if "stage_name" in df.columns:
        df["is_knockout"] = df["stage_name"].apply(_stage_knockout_flag).astype(int)
    elif "knockout_stage" in df.columns:
        df["is_knockout"] = to_num(df["knockout_stage"]).fillna(0).astype(int)
    else:
        df["is_knockout"] = 0

    return df.sort_values("match_date")


def add_last_features_old(matches, team_appearances):
    rows = []
    ta = team_appearances.copy()
    ta["match_date"] = parse_match_date(ta["match_date"])
    numeric_cols = ["win", "draw", "lose", "goals_for", "goals_against", "goal_differential"]
    for c in numeric_cols:
        if c in ta.columns:
            ta[c] = to_num(ta[c])

    def stats_before(team_id, date):
        hist = ta[(ta["team_id"] == team_id) & (ta["match_date"] < date)].sort_values("match_date")
        def calc(window):
            if window.empty:
                return {
                    "win_rate": 0.0, "draw_rate": 0.0, "loss_rate": 0.0,
                    "avg_goals_for": 0.0, "avg_goals_against": 0.0, "goal_diff": 0.0,
                    "points_per_match": 0.0, "matches_history": 0,
                }
            n = len(window)
            return {
                "win_rate": window["win"].fillna(0).sum() / n,
                "draw_rate": window["draw"].fillna(0).sum() / n,
                "loss_rate": window["lose"].fillna(0).sum() / n,
                "avg_goals_for": window["goals_for"].fillna(0).mean(),
                "avg_goals_against": window["goals_against"].fillna(0).mean(),
                "goal_diff": window["goal_differential"].fillna(0).mean(),
                "points_per_match": (3 * window["win"].fillna(0).sum() + window["draw"].fillna(0).sum()) / n,
                "matches_history": n,
            }
        last5 = calc(hist.tail(5))
        last10 = calc(hist.tail(10))
        allhist = calc(hist)
        out = {}
        for k, v in last5.items():
            out[f"{k}_last5"] = v
        for k, v in last10.items():
            out[f"{k}_last10"] = v
        for k, v in allhist.items():
            out[f"{k}_all"] = v
        return out

    for _, r in matches.iterrows():
        date = r["match_date"]
        home = stats_before(r["home_team_id"], date)
        away = stats_before(r["away_team_id"], date)
        item = r.to_dict()
        for k, v in home.items():
            item[f"home_{k}"] = v
        for k, v in away.items():
            item[f"away_{k}"] = v
        for suffix in ["last5", "last10", "all"]:
            for metric in ["win_rate", "draw_rate", "loss_rate", "avg_goals_for", "avg_goals_against", "goal_diff", "points_per_match"]:
                item[f"diff_{metric}_{suffix}"] = item.get(f"home_{metric}_{suffix}", 0) - item.get(f"away_{metric}_{suffix}", 0)
        rows.append(item)
    return pd.DataFrame(rows)


def add_team_context(df, teams):
    teams_small = teams[["team_id", "team_name", "confederation_id"]].copy()
    out = df.merge(teams_small.add_prefix("home_"), left_on="home_team_id", right_on="home_team_id", how="left")
    out = out.merge(teams_small.add_prefix("away_"), left_on="away_team_id", right_on="away_team_id", how="left")
    out["same_confederation"] = (out["home_confederation_id"] == out["away_confederation_id"]).astype(int)
    return out


def final_dataset_old(matches, team_appearances, teams):
    m = prepare_matches(matches)
    ta = team_appearances.merge(m[["match_id", "match_date"]], on="match_id", how="left")
    data = add_last_features_old(m, ta)
    data = add_team_context(data, teams)
    data["source_dataset"] = "old_three_table_dataset"
    keep = [
        "source_dataset", "match_id", "match_date", "tournament_id", "stage_name", "group_stage", "knockout_stage", "is_knockout",
        "home_team_id", "away_team_id", "home_team_name", "away_team_name", "home_confederation_id", "away_confederation_id", "same_confederation",
        "home_team_score", "away_team_score", "target", "target_name",
    ] + model_columns()
    keep_existing = []
    for c in keep:
        if c in data.columns and c not in keep_existing:
            keep_existing.append(c)
    return data[keep_existing].dropna(subset=["target"])


# -----------------------------------------------------------------------------
# New Kaggle-style dataset from data.zip: wc_matches_historical + wc_team_appearances
# -----------------------------------------------------------------------------

def _team_history_maps(appearances):
    app = appearances.copy()
    if app.empty:
        return {}, {}
    app["wc_year"] = to_num(app["wc_year"])
    for c in [
        "matches_played", "wins", "draws", "losses", "goals_scored", "goals_conceded",
        "goal_difference", "points_earned", "wc_titles_before_tournament",
        "consecutive_appearances", "elo_rating_approx", "fifa_ranking",
    ]:
        if c in app.columns:
            app[c] = to_num(app[c])
    if "participation_status" in app.columns:
        app = app[app["participation_status"].astype(str).str.lower().eq("qualified")].copy()
    team_hist = {team: g.sort_values("wc_year") for team, g in app.groupby("team")}
    conf_map = app.dropna(subset=["confederation"]).groupby("team")["confederation"].last().to_dict() if "confederation" in app.columns else {}
    return team_hist, conf_map


def _historical_team_features(team, wc_year, team_hist):
    hist = team_hist.get(team, pd.DataFrame()).copy()
    if len(hist):
        hist = hist[hist["wc_year"] < wc_year]
    if hist.empty:
        return {
            "hist_apps": 0.0, "hist_titles": 0.0, "hist_wins": 0.0,
            "hist_win_rate": 0.0, "hist_draw_rate": 0.0, "hist_loss_rate": 0.0,
            "hist_goal_diff_pm": 0.0, "hist_points_pm": 0.0,
            "hist_goals_for_pm": 0.0, "hist_goals_against_pm": 0.0,
            "hist_elo": 1600.0, "hist_fifa_rank": 100.0,
            "hist_years_since_last_wc": 20.0, "hist_consecutive_apps": 0.0,
            "pedigree_score": 0.0,
        }
    recent = hist.tail(4)
    mp = recent["matches_played"].fillna(0).sum()
    wins = recent["wins"].fillna(0).sum()
    draws = recent["draws"].fillna(0).sum()
    losses = recent["losses"].fillna(0).sum()
    gf = recent["goals_scored"].fillna(0).sum()
    ga = recent["goals_conceded"].fillna(0).sum()
    pts = recent["points_earned"].fillna(0).sum()
    apps = float(len(hist))
    last = hist.iloc[-1]
    titles = float(last.get("wc_titles_before_tournament", 0) if pd.notna(last.get("wc_titles_before_tournament", np.nan)) else 0)
    hist_wins_total = float(hist["wins"].fillna(0).sum()) if "wins" in hist else 0.0
    pedigree = titles * 3.0 + hist_wins_total * 0.1 + apps * 0.2
    return {
        "hist_apps": apps,
        "hist_titles": titles,
        "hist_wins": hist_wins_total,
        "hist_win_rate": _safe_div(wins, mp),
        "hist_draw_rate": _safe_div(draws, mp),
        "hist_loss_rate": _safe_div(losses, mp),
        "hist_goal_diff_pm": _safe_div(gf - ga, mp),
        "hist_points_pm": _safe_div(pts, mp),
        "hist_goals_for_pm": _safe_div(gf, mp),
        "hist_goals_against_pm": _safe_div(ga, mp),
        "hist_elo": float(last.get("elo_rating_approx", 1600) if pd.notna(last.get("elo_rating_approx", np.nan)) else 1600),
        "hist_fifa_rank": float(last.get("fifa_ranking", 100) if pd.notna(last.get("fifa_ranking", np.nan)) else 100),
        "hist_years_since_last_wc": float(wc_year - last.get("wc_year", wc_year)),
        "hist_consecutive_apps": float(last.get("consecutive_appearances", 0) if pd.notna(last.get("consecutive_appearances", np.nan)) else 0),
        "pedigree_score": pedigree,
    }


def final_dataset_kaggle(matches, appearances):
    m = matches.copy()
    m["match_date"] = parse_match_date(m["date"] if "date" in m.columns else m.get("match_date"))
    m["home_team_score"] = to_num(m["home_goals"] if "home_goals" in m.columns else m.get("home_team_score"))
    m["away_team_score"] = to_num(m["away_goals"] if "away_goals" in m.columns else m.get("away_team_score"))
    m["wc_year"] = to_num(m["wc_year"])
    m = m.dropna(subset=["match_date", "home_team_score", "away_team_score", "wc_year"])
    m["target"] = np.select(
        [m["home_team_score"] > m["away_team_score"], m["home_team_score"] == m["away_team_score"], m["home_team_score"] < m["away_team_score"]],
        [0, 1, 2], default=np.nan,
    )
    m = m.dropna(subset=["target"])
    m["target"] = m["target"].astype(int)
    m["target_name"] = m["target"].map(TARGET_LABELS)
    m["home_team_name"] = m["home_team"].astype(str).str.strip()
    m["away_team_name"] = m["away_team"].astype(str).str.strip()
    m["stage_name"] = m["stage"].astype(str)
    m["is_knockout"] = m["stage_name"].apply(_stage_knockout_flag).astype(int)
    m["home_elo"] = to_num(m.get("home_pre_match_elo", np.nan)).fillna(1600)
    m["away_elo"] = to_num(m.get("away_pre_match_elo", np.nan)).fillna(1600)
    m["elo_diff"] = m["home_elo"] - m["away_elo"]
    m["abs_elo_diff"] = m["elo_diff"].abs()
    m["home_is_elo_favorite"] = (m["elo_diff"] >= 0).astype(int)

    team_hist, conf_map = _team_history_maps(appearances)
    rows = []
    for _, r in m.iterrows():
        hname, aname, year = r["home_team_name"], r["away_team_name"], int(r["wc_year"])
        h = _historical_team_features(hname, year, team_hist)
        a = _historical_team_features(aname, year, team_hist)
        item = r.to_dict()
        item["home_confederation_id"] = conf_map.get(hname, "unknown")
        item["away_confederation_id"] = conf_map.get(aname, "unknown")
        item["same_confederation"] = int(item["home_confederation_id"] == item["away_confederation_id"])
        for k, v in h.items():
            item[f"home_{k}"] = v
        for k, v in a.items():
            item[f"away_{k}"] = v
        for metric in [
            "hist_apps", "hist_titles", "hist_wins", "hist_win_rate", "hist_draw_rate", "hist_loss_rate",
            "hist_goal_diff_pm", "hist_points_pm", "hist_goals_for_pm", "hist_goals_against_pm",
            "hist_elo", "hist_fifa_rank", "hist_years_since_last_wc", "hist_consecutive_apps", "pedigree_score",
        ]:
            item[f"diff_{metric}"] = item.get(f"home_{metric}", 0) - item.get(f"away_{metric}", 0)
        # Lower FIFA rank is better, so make a positive value mean home team is ranked better.
        item["fifa_rank_advantage"] = item.get("away_hist_fifa_rank", 100) - item.get("home_hist_fifa_rank", 100)
        item["source_dataset"] = "enhanced_kaggle_dataset"
        rows.append(item)
    data = pd.DataFrame(rows)
    keep = [
        "source_dataset", "match_id", "match_date", "wc_year", "stage_name", "is_knockout",
        "home_team_name", "away_team_name", "home_confederation_id", "away_confederation_id", "same_confederation",
        "home_team_score", "away_team_score", "target", "target_name",
    ] + model_columns()
    keep_existing = []
    for c in keep:
        if c in data.columns and c not in keep_existing:
            keep_existing.append(c)
    return data[keep_existing].dropna(subset=["target"])



# -----------------------------------------------------------------------------
# Hybrid local dataset: full old match set + safe enhanced historical features
# -----------------------------------------------------------------------------

def _norm_team_name(x):
    """Normalize common national-team aliases to the names used in the old match dataset.

    Keeping one canonical name is important when merging the old World Cup data,
    the Kaggle-style historical data, host data, and international match data.
    """
    s = str(x).strip()
    aliases = {
        "USA": "United States",
        "U.S.A.": "United States",
        "United States of America": "United States",
        "Korea Republic": "South Korea",
        "Republic of Korea": "South Korea",
        "Korea": "South Korea",
        "Korea DPR": "North Korea",
        "DPR Korea": "North Korea",
        "IR Iran": "Iran",
        "Islamic Republic of Iran": "Iran",
        "Germany FR": "West Germany",
        "Federal Republic of Germany": "West Germany",
        "Czechia": "Czech Republic",
        "Côte d'Ivoire": "Ivory Coast",
        "Cote d'Ivoire": "Ivory Coast",
        "Republic of Ireland": "Republic of Ireland",
        "Rep of Ireland": "Republic of Ireland",
        "Türkiye": "Turkey",
    }
    return aliases.get(s, s)


def _ratio(a, b, default=0.0):
    try:
        a = float(a)
        b = float(b)
        if pd.isna(a) or pd.isna(b) or abs(b) < 1e-9:
            return default
        return a / b
    except Exception:
        return default


def _pre_match_elo_lookup(enhanced_matches):
    if enhanced_matches is None or enhanced_matches.empty:
        return {}
    em = enhanced_matches.copy()
    em["home_goals"] = to_num(em.get("home_goals"))
    em["away_goals"] = to_num(em.get("away_goals"))
    lookup = {}
    for _, r in em.iterrows():
        try:
            year = int(r.get("wc_year"))
        except Exception:
            continue
        h = _norm_team_name(r.get("home_team"))
        a = _norm_team_name(r.get("away_team"))
        hg = r.get("home_goals")
        ag = r.get("away_goals")
        key = (year, h, a, int(hg) if pd.notna(hg) else None, int(ag) if pd.notna(ag) else None)
        lookup[key] = (r.get("home_pre_match_elo", np.nan), r.get("away_pre_match_elo", np.nan))
        # Also store reversed orientation in case old data has opposite home/away order.
        key_rev = (year, a, h, int(ag) if pd.notna(ag) else None, int(hg) if pd.notna(hg) else None)
        lookup[key_rev] = (r.get("away_pre_match_elo", np.nan), r.get("home_pre_match_elo", np.nan))
    return lookup


def _add_enhanced_diffs(item):
    # Core differences and ratios. Positive values generally mean home team advantage.
    item["elo_diff"] = item.get("home_elo", 1600) - item.get("away_elo", 1600)
    item["abs_elo_diff"] = abs(item["elo_diff"])
    item["elo_ratio"] = _ratio(item.get("home_elo", 1600), item.get("away_elo", 1600), 1.0)
    item["home_is_elo_favorite"] = int(item["elo_diff"] >= 0)
    item["fifa_rank_advantage"] = item.get("away_hist_fifa_rank", 100) - item.get("home_hist_fifa_rank", 100)
    item["experience_diff"] = item.get("home_hist_apps", 0) - item.get("away_hist_apps", 0)
    item["titles_diff"] = item.get("home_hist_titles", 0) - item.get("away_hist_titles", 0)
    item["points_per_match_diff"] = item.get("home_hist_points_pm", 0) - item.get("away_hist_points_pm", 0)
    item["goal_diff_diff"] = item.get("home_hist_goal_diff_pm", 0) - item.get("away_hist_goal_diff_pm", 0)
    item["form_diff"] = item.get("home_hist_points_pm", 0) - item.get("away_hist_points_pm", 0)
    item["hist_apps_ratio"] = _ratio(1 + item.get("home_hist_apps", 0), 1 + item.get("away_hist_apps", 0), 1.0)
    item["hist_titles_ratio"] = _ratio(1 + item.get("home_hist_titles", 0), 1 + item.get("away_hist_titles", 0), 1.0)
    item["hist_points_pm_ratio"] = _ratio(1 + item.get("home_hist_points_pm", 0), 1 + item.get("away_hist_points_pm", 0), 1.0)
    item["hist_goal_diff_pm_ratio"] = _ratio(3 + item.get("home_hist_goal_diff_pm", 0), 3 + item.get("away_hist_goal_diff_pm", 0), 1.0)
    item["favorite_win_binary"] = int(
        (item.get("home_is_elo_favorite", 0) == 1 and item.get("target") == 0)
        or (item.get("home_is_elo_favorite", 0) == 0 and item.get("target") == 2)
    )
    return item




def _build_international_history(international_matches):
    """Return team -> chronological international-match history.

    The table contains matches outside and inside World Cup. For every target match,
    downstream code uses only Date < match_date, so the current match and future matches
    cannot leak into the feature values.
    """
    if international_matches is None or international_matches.empty:
        return {}
    im = international_matches.copy()
    date_col = "Date" if "Date" in im.columns else "date"
    home_col = "Home Team" if "Home Team" in im.columns else "home_team"
    away_col = "Away Team" if "Away Team" in im.columns else "away_team"
    hg_col = "Home Goals" if "Home Goals" in im.columns else "home_goals"
    ag_col = "Away Goals" if "Away Goals" in im.columns else "away_goals"
    tourn_col = "Tournament" if "Tournament" in im.columns else "tournament"

    im[date_col] = parse_match_date(im[date_col])
    im[hg_col] = to_num(im[hg_col])
    im[ag_col] = to_num(im[ag_col])
    im = im.dropna(subset=[date_col, home_col, away_col, hg_col, ag_col])

    rows = []
    for _, r in im.iterrows():
        date = r[date_col]
        h = _norm_team_name(r[home_col])
        a = _norm_team_name(r[away_col])
        hg = float(r[hg_col])
        ag = float(r[ag_col])
        tournament = str(r.get(tourn_col, ""))
        is_comp = int(tournament.lower().strip() != "friendly")
        rows.append({
            "team": h, "date": date, "opponent": a, "goals_for": hg, "goals_against": ag,
            "win": int(hg > ag), "draw": int(hg == ag), "loss": int(hg < ag),
            "clean_sheet": int(ag == 0), "unbeaten": int(hg >= ag), "competitive": is_comp,
        })
        rows.append({
            "team": a, "date": date, "opponent": h, "goals_for": ag, "goals_against": hg,
            "win": int(ag > hg), "draw": int(ag == hg), "loss": int(ag < hg),
            "clean_sheet": int(hg == 0), "unbeaten": int(ag >= hg), "competitive": is_comp,
        })
    long = pd.DataFrame(rows)
    if long.empty:
        return {}
    return {team: g.sort_values("date") for team, g in long.groupby("team")}


def _recent_intl_features(team, match_date, intl_hist, window=10, competitive_only=False):
    prefix = "comp" if competitive_only else "intl"
    defaults = {
        f"recent_{prefix}_matches_last{window}": 0.0,
        f"recent_{prefix}_win_rate_last{window}": 0.0,
        f"recent_{prefix}_draw_rate_last{window}": 0.0,
        f"recent_{prefix}_loss_rate_last{window}": 0.0,
        f"recent_{prefix}_goals_for_last{window}": 0.0,
        f"recent_{prefix}_goals_against_last{window}": 0.0,
        f"recent_{prefix}_goal_diff_last{window}": 0.0,
        f"recent_{prefix}_points_pm_last{window}": 0.0,
        f"recent_{prefix}_unbeaten_rate_last{window}": 0.0,
        f"recent_{prefix}_clean_sheet_rate_last{window}": 0.0,
        f"recent_{prefix}_days_since_last_match": 3650.0,
    }
    hist = intl_hist.get(_norm_team_name(team), pd.DataFrame()).copy()
    if hist.empty or pd.isna(match_date):
        return defaults
    hist = hist[hist["date"] < match_date].copy()
    if competitive_only:
        hist = hist[hist["competitive"] == 1]
    hist = hist.sort_values("date").tail(window)
    if hist.empty:
        return defaults
    n = float(len(hist))
    gf = hist["goals_for"].fillna(0).mean()
    ga = hist["goals_against"].fillna(0).mean()
    wins = hist["win"].fillna(0).sum()
    draws = hist["draw"].fillna(0).sum()
    last_date = hist["date"].max()
    return {
        f"recent_{prefix}_matches_last{window}": n,
        f"recent_{prefix}_win_rate_last{window}": _safe_div(wins, n),
        f"recent_{prefix}_draw_rate_last{window}": _safe_div(draws, n),
        f"recent_{prefix}_loss_rate_last{window}": _safe_div(hist["loss"].fillna(0).sum(), n),
        f"recent_{prefix}_goals_for_last{window}": float(gf),
        f"recent_{prefix}_goals_against_last{window}": float(ga),
        f"recent_{prefix}_goal_diff_last{window}": float(gf - ga),
        f"recent_{prefix}_points_pm_last{window}": _safe_div(3 * wins + draws, n),
        f"recent_{prefix}_unbeaten_rate_last{window}": _safe_div(hist["unbeaten"].fillna(0).sum(), n),
        f"recent_{prefix}_clean_sheet_rate_last{window}": _safe_div(hist["clean_sheet"].fillna(0).sum(), n),
        f"recent_{prefix}_days_since_last_match": float((match_date - last_date).days) if pd.notna(last_date) else 3650.0,
    }


def _build_host_map(world_cup_host):
    if world_cup_host is None or world_cup_host.empty:
        return {}
    h = world_cup_host.copy()
    year_col = "Year" if "Year" in h.columns else "wc_year"
    host_col = "Host Country" if "Host Country" in h.columns else "host"
    out = {}
    for _, r in h.iterrows():
        try:
            year = int(r[year_col])
        except Exception:
            continue
        raw = str(r.get(host_col, ""))
        # Handles values such as Korea/Japan and comma-separated co-hosts.
        parts = re.split(r"/|,|;| and ", raw)
        hosts = {_norm_team_name(p) for p in parts if str(p).strip()}
        out[year] = hosts
    return out


def _build_standings_history(tournament_standings):
    if tournament_standings is None or tournament_standings.empty:
        return {}
    stn = tournament_standings.copy()
    if "tournament_id" not in stn.columns or "team_id" not in stn.columns or "position" not in stn.columns:
        return {}
    stn["wc_year"] = stn["tournament_id"].astype(str).str.extract(r"(\d{4})")[0].pipe(to_num)
    stn["position"] = to_num(stn["position"])
    stn = stn.dropna(subset=["wc_year", "team_id", "position"])
    return {team: g.sort_values("wc_year") for team, g in stn.groupby("team_id")}


def _standing_features(team_id, wc_year, standing_hist):
    hist = standing_hist.get(team_id, pd.DataFrame()).copy()
    if len(hist):
        hist = hist[hist["wc_year"] < wc_year]
    if hist.empty:
        return {
            "standing_best_position_before": 33.0,
            "standing_avg_position_before": 33.0,
            "standing_top4_count_before": 0.0,
            "standing_top8_count_before": 0.0,
            "standing_champion_count_before": 0.0,
        }
    pos = hist["position"].astype(float)
    return {
        "standing_best_position_before": float(pos.min()),
        "standing_avg_position_before": float(pos.mean()),
        "standing_top4_count_before": float((pos <= 4).sum()),
        "standing_top8_count_before": float((pos <= 8).sum()),
        "standing_champion_count_before": float((pos == 1).sum()),
    }

def final_dataset_hybrid(old_matches, old_team_appearances, old_teams, enhanced_matches=None, enhanced_appearances=None, international_matches=None, world_cup_host=None, tournament_standings=None):
    """Build the main clean dataset used by the final app.

    It keeps the complete old match-level World Cup data, then adds safe enhanced features
    computed from historical World Cup appearances before the current tournament year.
    2026 snapshot variables such as market value, coach information, and prediction probability
    are intentionally not used for 1930-2022 validation to avoid future-information leakage.
    """
    base = final_dataset_old(old_matches, old_team_appearances, old_teams).copy()
    base["match_date"] = pd.to_datetime(base["match_date"], errors="coerce")
    base["wc_year"] = base["match_date"].dt.year

    if enhanced_appearances is None or enhanced_appearances.empty:
        team_hist, conf_map = {}, {}
    else:
        ea = enhanced_appearances.copy()
        ea["team"] = ea["team"].map(_norm_team_name)
        team_hist, conf_map = _team_history_maps(ea)

    elo_lookup = _pre_match_elo_lookup(enhanced_matches if enhanced_matches is not None else pd.DataFrame())
    intl_hist = _build_international_history(international_matches if international_matches is not None else pd.DataFrame())
    host_map = _build_host_map(world_cup_host if world_cup_host is not None else pd.DataFrame())
    standing_hist = _build_standings_history(tournament_standings if tournament_standings is not None else pd.DataFrame())
    rows = []
    for _, r in base.iterrows():
        item = r.to_dict()
        year = int(item.get("wc_year")) if pd.notna(item.get("wc_year")) else 0
        hname = _norm_team_name(item.get("home_team_name"))
        aname = _norm_team_name(item.get("away_team_name"))
        h = _historical_team_features(hname, year, team_hist)
        a = _historical_team_features(aname, year, team_hist)
        for k, v in h.items():
            item[f"home_{k}"] = v
        for k, v in a.items():
            item[f"away_{k}"] = v
        hg = int(item.get("home_team_score")) if pd.notna(item.get("home_team_score")) else None
        ag = int(item.get("away_team_score")) if pd.notna(item.get("away_team_score")) else None
        ehome, eaway = elo_lookup.get((year, hname, aname, hg, ag), (np.nan, np.nan))
        item["home_elo"] = float(ehome) if pd.notna(ehome) else float(item.get("home_hist_elo", 1600))
        item["away_elo"] = float(eaway) if pd.notna(eaway) else float(item.get("away_hist_elo", 1600))

        # Recent international form: computed strictly from matches before current match_date.
        match_date = item.get("match_date")
        for k, v in _recent_intl_features(hname, match_date, intl_hist, window=10, competitive_only=False).items():
            item[f"home_{k}"] = v
        for k, v in _recent_intl_features(aname, match_date, intl_hist, window=10, competitive_only=False).items():
            item[f"away_{k}"] = v
        for k, v in _recent_intl_features(hname, match_date, intl_hist, window=10, competitive_only=True).items():
            item[f"home_{k}"] = v
        for k, v in _recent_intl_features(aname, match_date, intl_hist, window=10, competitive_only=True).items():
            item[f"away_{k}"] = v

        # Host advantage: known before the tournament; no result leakage.
        hosts = host_map.get(year, set())
        item["home_is_host"] = int(hname in hosts)
        item["away_is_host"] = int(aname in hosts)
        item["host_advantage"] = item["home_is_host"] - item["away_is_host"]

        # Historical final-standing features: only standings before current WC year.
        hs = _standing_features(item.get("home_team_id"), year, standing_hist)
        aw = _standing_features(item.get("away_team_id"), year, standing_hist)
        for k, v in hs.items():
            item[f"home_{k}"] = v
        for k, v in aw.items():
            item[f"away_{k}"] = v

        for metric in [
            "hist_apps", "hist_titles", "hist_wins", "hist_win_rate", "hist_draw_rate", "hist_loss_rate",
            "hist_goal_diff_pm", "hist_points_pm", "hist_goals_for_pm", "hist_goals_against_pm",
            "hist_elo", "hist_fifa_rank", "hist_years_since_last_wc", "hist_consecutive_apps", "pedigree_score",
        ]:
            item[f"diff_{metric}"] = item.get(f"home_{metric}", 0) - item.get(f"away_{metric}", 0)
        # Differences for new recent-form, host, and standings features.
        for metric in [
            "recent_intl_matches_last10", "recent_intl_win_rate_last10", "recent_intl_draw_rate_last10",
            "recent_intl_loss_rate_last10", "recent_intl_goals_for_last10", "recent_intl_goals_against_last10",
            "recent_intl_goal_diff_last10", "recent_intl_points_pm_last10", "recent_intl_unbeaten_rate_last10",
            "recent_intl_clean_sheet_rate_last10", "recent_intl_days_since_last_match",
            "recent_comp_matches_last10", "recent_comp_win_rate_last10", "recent_comp_draw_rate_last10",
            "recent_comp_loss_rate_last10", "recent_comp_goals_for_last10", "recent_comp_goals_against_last10",
            "recent_comp_goal_diff_last10", "recent_comp_points_pm_last10", "recent_comp_unbeaten_rate_last10",
            "recent_comp_clean_sheet_rate_last10", "recent_comp_days_since_last_match",
            "standing_best_position_before", "standing_avg_position_before", "standing_top4_count_before",
            "standing_top8_count_before", "standing_champion_count_before",
        ]:
            item[f"diff_{metric}"] = item.get(f"home_{metric}", 0) - item.get(f"away_{metric}", 0)
        # For standing position, lower is better, so create a positive home advantage.
        item["standing_best_position_advantage"] = item.get("away_standing_best_position_before", 33) - item.get("home_standing_best_position_before", 33)
        item["standing_avg_position_advantage"] = item.get("away_standing_avg_position_before", 33) - item.get("home_standing_avg_position_before", 33)

        item = _add_enhanced_diffs(item)
        item["source_dataset"] = "hybrid_full_old_plus_safe_enhanced_intl_host_standing_features"
        rows.append(item)
    data = pd.DataFrame(rows)
    keep = [
        "source_dataset", "match_id", "match_date", "wc_year", "tournament_id", "stage_name", "is_knockout",
        "home_team_id", "away_team_id", "home_team_name", "away_team_name",
        "home_confederation_id", "away_confederation_id", "same_confederation",
        "home_team_score", "away_team_score", "target", "target_name", "favorite_win_binary",
    ] + model_columns()
    keep_existing = []
    for c in keep:
        if c in data.columns and c not in keep_existing:
            keep_existing.append(c)
    return data[keep_existing].dropna(subset=["target"]).reset_index(drop=True)


def final_dataset(matches, team_appearances, teams=None, enhanced_matches=None, enhanced_appearances=None, international_matches=None, world_cup_host=None, tournament_standings=None):
    # New data.zip structure only
    if "wc_year" in matches.columns and "home_team" in matches.columns and "home_goals" in matches.columns:
        return final_dataset_kaggle(matches, team_appearances)
    # Hybrid: complete old match set + safe enhanced historical features.
    if teams is None:
        raise ValueError("Old three-table dataset needs teams table.")
    if enhanced_matches is not None or enhanced_appearances is not None:
        return final_dataset_hybrid(matches, team_appearances, teams, enhanced_matches, enhanced_appearances, international_matches, world_cup_host, tournament_standings)
    return final_dataset_old(matches, team_appearances, teams)


def model_columns():
    legacy = [
        "same_confederation", "is_knockout",
        "home_win_rate_last5", "home_draw_rate_last5", "home_loss_rate_last5",
        "home_avg_goals_for_last5", "home_avg_goals_against_last5", "home_goal_diff_last5", "home_points_per_match_last5",
        "away_win_rate_last5", "away_draw_rate_last5", "away_loss_rate_last5",
        "away_avg_goals_for_last5", "away_avg_goals_against_last5", "away_goal_diff_last5", "away_points_per_match_last5",
        "diff_win_rate_last5", "diff_draw_rate_last5", "diff_loss_rate_last5", "diff_goal_diff_last5",
        "diff_avg_goals_for_last5", "diff_avg_goals_against_last5", "diff_points_per_match_last5",
        "home_win_rate_last10", "home_goal_diff_last10", "home_points_per_match_last10",
        "away_win_rate_last10", "away_goal_diff_last10", "away_points_per_match_last10",
        "diff_win_rate_last10", "diff_goal_diff_last10", "diff_points_per_match_last10",
        "home_win_rate_all", "home_goal_diff_all", "home_points_per_match_all", "home_matches_history_all",
        "away_win_rate_all", "away_goal_diff_all", "away_points_per_match_all", "away_matches_history_all",
        "diff_win_rate_all", "diff_goal_diff_all", "diff_points_per_match_all",
    ]
    enhanced = [
        "home_elo", "away_elo", "elo_diff", "abs_elo_diff", "elo_ratio", "home_is_elo_favorite",
        "home_hist_apps", "away_hist_apps", "diff_hist_apps", "experience_diff", "hist_apps_ratio",
        "home_hist_titles", "away_hist_titles", "diff_hist_titles", "titles_diff", "hist_titles_ratio",
        "home_hist_wins", "away_hist_wins", "diff_hist_wins",
        "home_hist_win_rate", "away_hist_win_rate", "diff_hist_win_rate",
        "home_hist_draw_rate", "away_hist_draw_rate", "diff_hist_draw_rate",
        "home_hist_loss_rate", "away_hist_loss_rate", "diff_hist_loss_rate",
        "home_hist_goal_diff_pm", "away_hist_goal_diff_pm", "diff_hist_goal_diff_pm", "goal_diff_diff", "hist_goal_diff_pm_ratio",
        "home_hist_points_pm", "away_hist_points_pm", "diff_hist_points_pm", "points_per_match_diff", "form_diff", "hist_points_pm_ratio",
        "home_hist_goals_for_pm", "away_hist_goals_for_pm", "diff_hist_goals_for_pm",
        "home_hist_goals_against_pm", "away_hist_goals_against_pm", "diff_hist_goals_against_pm",
        "home_hist_elo", "away_hist_elo", "diff_hist_elo",
        "home_hist_fifa_rank", "away_hist_fifa_rank", "diff_hist_fifa_rank", "fifa_rank_advantage",
        "home_hist_years_since_last_wc", "away_hist_years_since_last_wc", "diff_hist_years_since_last_wc",
        "home_hist_consecutive_apps", "away_hist_consecutive_apps", "diff_hist_consecutive_apps",
        "home_pedigree_score", "away_pedigree_score", "diff_pedigree_score",
    ]
    recent_form = [
        "home_recent_intl_matches_last10", "away_recent_intl_matches_last10", "diff_recent_intl_matches_last10",
        "home_recent_intl_win_rate_last10", "away_recent_intl_win_rate_last10", "diff_recent_intl_win_rate_last10",
        "home_recent_intl_draw_rate_last10", "away_recent_intl_draw_rate_last10", "diff_recent_intl_draw_rate_last10",
        "home_recent_intl_loss_rate_last10", "away_recent_intl_loss_rate_last10", "diff_recent_intl_loss_rate_last10",
        "home_recent_intl_goals_for_last10", "away_recent_intl_goals_for_last10", "diff_recent_intl_goals_for_last10",
        "home_recent_intl_goals_against_last10", "away_recent_intl_goals_against_last10", "diff_recent_intl_goals_against_last10",
        "home_recent_intl_goal_diff_last10", "away_recent_intl_goal_diff_last10", "diff_recent_intl_goal_diff_last10",
        "home_recent_intl_points_pm_last10", "away_recent_intl_points_pm_last10", "diff_recent_intl_points_pm_last10",
        "home_recent_intl_unbeaten_rate_last10", "away_recent_intl_unbeaten_rate_last10", "diff_recent_intl_unbeaten_rate_last10",
        "home_recent_intl_clean_sheet_rate_last10", "away_recent_intl_clean_sheet_rate_last10", "diff_recent_intl_clean_sheet_rate_last10",
        "home_recent_intl_days_since_last_match", "away_recent_intl_days_since_last_match", "diff_recent_intl_days_since_last_match",
        "home_recent_comp_matches_last10", "away_recent_comp_matches_last10", "diff_recent_comp_matches_last10",
        "home_recent_comp_win_rate_last10", "away_recent_comp_win_rate_last10", "diff_recent_comp_win_rate_last10",
        "home_recent_comp_draw_rate_last10", "away_recent_comp_draw_rate_last10", "diff_recent_comp_draw_rate_last10",
        "home_recent_comp_loss_rate_last10", "away_recent_comp_loss_rate_last10", "diff_recent_comp_loss_rate_last10",
        "home_recent_comp_goals_for_last10", "away_recent_comp_goals_for_last10", "diff_recent_comp_goals_for_last10",
        "home_recent_comp_goals_against_last10", "away_recent_comp_goals_against_last10", "diff_recent_comp_goals_against_last10",
        "home_recent_comp_goal_diff_last10", "away_recent_comp_goal_diff_last10", "diff_recent_comp_goal_diff_last10",
        "home_recent_comp_points_pm_last10", "away_recent_comp_points_pm_last10", "diff_recent_comp_points_pm_last10",
        "home_recent_comp_unbeaten_rate_last10", "away_recent_comp_unbeaten_rate_last10", "diff_recent_comp_unbeaten_rate_last10",
        "home_recent_comp_clean_sheet_rate_last10", "away_recent_comp_clean_sheet_rate_last10", "diff_recent_comp_clean_sheet_rate_last10",
        "home_recent_comp_days_since_last_match", "away_recent_comp_days_since_last_match", "diff_recent_comp_days_since_last_match",
    ]
    host_and_standing = [
        "home_is_host", "away_is_host", "host_advantage",
        "home_standing_best_position_before", "away_standing_best_position_before", "diff_standing_best_position_before", "standing_best_position_advantage",
        "home_standing_avg_position_before", "away_standing_avg_position_before", "diff_standing_avg_position_before", "standing_avg_position_advantage",
        "home_standing_top4_count_before", "away_standing_top4_count_before", "diff_standing_top4_count_before",
        "home_standing_top8_count_before", "away_standing_top8_count_before", "diff_standing_top8_count_before",
        "home_standing_champion_count_before", "away_standing_champion_count_before", "diff_standing_champion_count_before",
    ]

    seen = []
    for c in legacy + enhanced + recent_form + host_and_standing:
        if c not in seen:
            seen.append(c)
    return seen
