import os
import numpy as np
import pandas as pd


def _read_optional(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def load_2026_bundle(base_dir):
    enh = os.path.join(base_dir, "data", "enhanced")
    feat = _read_optional(os.path.join(enh, "wc_prediction_features_2026.csv"))
    groups = _read_optional(os.path.join(enh, "wc_2026_groups.csv"))
    qual = _read_optional(os.path.join(enh, "wc_2026_qualifying_summary.csv"))
    diff = _read_optional(os.path.join(enh, "wc_2026_group_difficulty.csv"))
    coaches = _read_optional(os.path.join(enh, "wc_coaches_2026.csv"))
    df = feat.copy()
    if df.empty and not groups.empty:
        df = groups.copy()
    for extra in [groups, qual, diff, coaches]:
        if not extra.empty and "team" in extra.columns and not df.empty:
            keep = [c for c in extra.columns if c == "team" or c not in df.columns]
            df = df.merge(extra[keep], on="team", how="left")
    return df


def _num(row, *names, default=0.0):
    for n in names:
        if n in row.index:
            try:
                v = float(row[n])
                if np.isfinite(v):
                    return v
            except Exception:
                pass
    return float(default)


def build_match_features_2026(home_team, away_team, teams2026, selected_features):
    if teams2026.empty:
        raise ValueError("Không có dữ liệu 2026.")
    t = teams2026.set_index("team")
    h = t.loc[home_team]
    a = t.loc[away_team]
    item = {}
    # Strength / rank
    item["home_elo"] = _num(h, "elo_rating_2026", "elo_rating", default=1600)
    item["away_elo"] = _num(a, "elo_rating_2026", "elo_rating", default=1600)
    item["elo_diff"] = item["home_elo"] - item["away_elo"]
    item["abs_elo_diff"] = abs(item["elo_diff"])
    item["elo_ratio"] = item["home_elo"] / max(item["away_elo"], 1)
    item["home_is_elo_favorite"] = int(item["home_elo"] >= item["away_elo"])
    hr = _num(h, "fifa_rank_apr2026", "fifa_rank", default=100)
    ar = _num(a, "fifa_rank_apr2026", "fifa_rank", default=100)
    item["home_hist_fifa_rank"] = hr
    item["away_hist_fifa_rank"] = ar
    item["diff_hist_fifa_rank"] = hr - ar
    item["fifa_rank_advantage"] = ar - hr  # bigger is better for home
    # WC history
    h_apps = _num(h, "wc_appearances", "wc_appearances_before_2026", default=0)
    a_apps = _num(a, "wc_appearances", "wc_appearances_before_2026", default=0)
    h_titles = _num(h, "titles_before_2026", "wc_titles", default=0)
    a_titles = _num(a, "titles_before_2026", "wc_titles", default=0)
    item["home_hist_apps"] = h_apps; item["away_hist_apps"] = a_apps; item["diff_hist_apps"] = h_apps - a_apps; item["experience_diff"] = h_apps - a_apps
    item["home_hist_titles"] = h_titles; item["away_hist_titles"] = a_titles; item["diff_hist_titles"] = h_titles - a_titles; item["titles_diff"] = h_titles - a_titles
    item["home_pedigree_score"] = h_titles * 3 + h_apps * 0.2
    item["away_pedigree_score"] = a_titles * 3 + a_apps * 0.2
    item["diff_pedigree_score"] = item["home_pedigree_score"] - item["away_pedigree_score"]
    # Historical performance proxies from available 2026 pre-tournament fields.
    h_best = _num(h, "best_finish_encoded", default=1)
    a_best = _num(a, "best_finish_encoded", default=1)
    # In training, lower standing position is better; convert encoded finish to a rough position proxy.
    h_pos = max(1.0, 9.0 - h_best)
    a_pos = max(1.0, 9.0 - a_best)
    item["home_standing_best_position_before"] = h_pos
    item["away_standing_best_position_before"] = a_pos
    item["diff_standing_best_position_before"] = h_pos - a_pos
    item["standing_best_position_advantage"] = a_pos - h_pos
    item["home_standing_avg_position_before"] = h_pos
    item["away_standing_avg_position_before"] = a_pos
    item["diff_standing_avg_position_before"] = h_pos - a_pos
    item["standing_avg_position_advantage"] = a_pos - h_pos
    item["home_standing_champion_count_before"] = h_titles
    item["away_standing_champion_count_before"] = a_titles
    item["diff_standing_champion_count_before"] = h_titles - a_titles
    item["home_hist_win_rate"] = _num(h, "qualifying_win_rate", default=0.0)
    item["away_hist_win_rate"] = _num(a, "qualifying_win_rate", default=0.0)
    item["diff_hist_win_rate"] = item["home_hist_win_rate"] - item["away_hist_win_rate"]
    # 2026 recent/qualifying proxies mapped to train-time recent form names
    h_recent = _num(h, "recent_form_pts_last10", default=0)
    a_recent = _num(a, "recent_form_pts_last10", default=0)
    item["home_recent_intl_points_pm_last10"] = h_recent / 10.0
    item["away_recent_intl_points_pm_last10"] = a_recent / 10.0
    item["diff_recent_intl_points_pm_last10"] = item["home_recent_intl_points_pm_last10"] - item["away_recent_intl_points_pm_last10"]
    item["home_recent_comp_points_pm_last10"] = item["home_recent_intl_points_pm_last10"]
    item["away_recent_comp_points_pm_last10"] = item["away_recent_intl_points_pm_last10"]
    item["diff_recent_comp_points_pm_last10"] = item["diff_recent_intl_points_pm_last10"]
    hq_gf = _num(h, "qualifying_gf", default=0); aq_gf = _num(a, "qualifying_gf", default=0)
    hq_ga = _num(h, "qualifying_ga", default=0); aq_ga = _num(a, "qualifying_ga", default=0)
    hq_played = max(_num(h, "qualifying_played", default=10), 1); aq_played = max(_num(a, "qualifying_played", default=10), 1)
    item["home_recent_comp_goals_for_last10"] = hq_gf / hq_played
    item["away_recent_comp_goals_for_last10"] = aq_gf / aq_played
    item["diff_recent_comp_goals_for_last10"] = item["home_recent_comp_goals_for_last10"] - item["away_recent_comp_goals_for_last10"]
    item["home_recent_comp_goals_against_last10"] = hq_ga / hq_played
    item["away_recent_comp_goals_against_last10"] = aq_ga / aq_played
    item["diff_recent_comp_goals_against_last10"] = item["home_recent_comp_goals_against_last10"] - item["away_recent_comp_goals_against_last10"]
    item["home_recent_comp_goal_diff_last10"] = (hq_gf - hq_ga) / hq_played
    item["away_recent_comp_goal_diff_last10"] = (aq_gf - aq_ga) / aq_played
    item["diff_recent_comp_goal_diff_last10"] = item["home_recent_comp_goal_diff_last10"] - item["away_recent_comp_goal_diff_last10"]
    item["home_hist_goal_diff_pm"] = item["home_recent_comp_goal_diff_last10"]
    item["away_hist_goal_diff_pm"] = item["away_recent_comp_goal_diff_last10"]
    item["diff_hist_goal_diff_pm"] = item["diff_recent_comp_goal_diff_last10"]
    item["goal_diff_diff"] = item["diff_recent_comp_goal_diff_last10"]
    item["home_hist_points_pm"] = _num(h, "qualifying_pts", default=h_recent) / hq_played
    item["away_hist_points_pm"] = _num(a, "qualifying_pts", default=a_recent) / aq_played
    item["diff_hist_points_pm"] = item["home_hist_points_pm"] - item["away_hist_points_pm"]
    item["points_per_match_diff"] = item["diff_hist_points_pm"]
    item["form_diff"] = item["diff_hist_points_pm"]
    hwr = _num(h, "qualifying_win_rate", default=0); awr = _num(a, "qualifying_win_rate", default=0)
    item["home_recent_comp_win_rate_last10"] = hwr
    item["away_recent_comp_win_rate_last10"] = awr
    item["diff_recent_comp_win_rate_last10"] = hwr - awr
    # Market and coach values are 2026-only; use only if selected feature happens to exist.
    item["market_value_ratio"] = _num(h, "squad_market_value_eur_m", "squad_market_value_eur_millions", default=0) / max(_num(a, "squad_market_value_eur_m", "squad_market_value_eur_millions", default=1), 1)
    item["coach_exp_diff"] = _num(h, "coach_wc_experience", "wc_appearances_as_coach", default=0) - _num(a, "coach_wc_experience", "wc_appearances_as_coach", default=0)
    # Context
    home_host = str(h.get("host_nation", h.get("is_host", "No"))).lower() in ["yes", "true", "1"]
    away_host = str(a.get("host_nation", a.get("is_host", "No"))).lower() in ["yes", "true", "1"]
    item["home_is_host"] = int(home_host)
    item["away_is_host"] = int(away_host)
    item["host_advantage"] = item["home_is_host"] - item["away_is_host"]
    item["same_confederation"] = int(str(h.get("confederation", "")) == str(a.get("confederation", "")))
    item["is_knockout"] = 0
    # Return in selected feature order, missing filled 0.
    return pd.DataFrame([{f: float(item.get(f, 0.0)) for f in selected_features}])
