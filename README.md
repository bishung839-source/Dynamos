import pandas as pd
import numpy as np
from scipy.stats import poisson

class AdvancedMatchPredictor:
    def __init__(self, league_avg_goals=1.45):
        """
        Initialize the model with league context.
        :param league_avg_goals: Average goals per game in the league (Ligue 1 approx 2.9 total, so 1.45 per team).
        """
        self.league_avg = league_avg_goals
        self.team_stats = {}

    def calculate_strength(self, team_name, goals_for, goals_against, games_played, is_home):
        """
        Calculates the raw Attack and Defense strengths based on historical performance.
        Strengths are relative to the league average.
        """
        avg_scored = goals_for / games_played
        avg_conceded = goals_against / games_played

        # Attack Strength = Team's Avg Scored / League Avg
        attack_strength = avg_scored / self.league_avg
        
        # Defense Strength = Team's Avg Conceded / League Avg
        # (Lower is better for defense, but we keep it as a ratio for multiplication)
        defense_strength = avg_conceded / self.league_avg

        self.team_stats[team_name] = {
            'attack': attack_strength,
            'defense': defense_strength,
            'is_home_stats': is_home 
        }
        return self.team_stats[team_name]

    def apply_alpha_adjustments(self, home_team, away_team, factors):
        """
        Applies situational 'Alpha' factors (Injuries, Fatigue, Motivation).
        Returns adjusted Expected Goals (xG) for both teams.
        """
        # 1. Get Base Strengths
        h_stats = self.team_stats.get(home_team)
        a_stats = self.team_stats.get(away_team)

        if not h_stats or not a_stats:
            raise ValueError("Teams not found in database. Run calculate_strength first.")

        # 2. Calculate Base Expected Goals (The Standard Poisson Formula)
        # Home Goals = Home Attack * Away Defense * League Avg
        home_lambda = h_stats['attack'] * a_stats['defense'] * self.league_avg
        
        # Away Goals = Away Attack * Home Defense * League Avg
        away_lambda = a_stats['attack'] * h_stats['defense'] * self.league_avg

        print(f"\n--- Base Metrics (Before Adjustments) ---")
        print(f"{home_team} Base xG: {home_lambda:.2f}")
        print(f"{away_team} Base xG: {away_lambda:.2f}")

        # 3. Apply Adjustments (The "Alpha")
        # Factors should be a dictionary like: {'home_injury_penalty': 0.90, 'away_fatigue': 0.95}
        
        # Adjust Home Team
        if 'home_missing_key_player' in factors:
            print(f"Applying penalty to {home_team} for missing players...")
            home_lambda *= factors['home_missing_key_player'] # e.g., multiply by 0.85 (15% drop)
            
        if 'home_fatigue' in factors:
            print(f"Applying fatigue penalty to {home_team}...")
            home_lambda *= factors['home_fatigue']

        # Adjust Away Team
        if 'away_missing_key_player' in factors:
            print(f"Applying penalty to {away_team} for missing players...")
            away_lambda *= factors['away_missing_key_player']
            
        if 'away_h2h_mental_block' in factors:
            # If Away team struggles historically against Home team
            print(f"Applying H2H mental block penalty to {away_team}...")
            away_lambda *= factors['away_h2h_mental_block']

        return home_lambda, away_lambda

    def simulate_match(self, home_xg, away_xg, max_goals=10):
        """
        Runs the Poisson simulation to find Win/Draw/Loss probabilities.
        """
        home_win_prob = 0
        draw_prob = 0
        away_win_prob = 0

        # Matrix calculation for every scoreline from 0-0 to 9-9
        for h in range(max_goals):
            for a in range(max_goals):
                prob = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
                
                if h > a:
                    home_win_prob += prob
                elif h == a:
                    draw_prob += prob
                else:
                    away_win_prob += prob

        return home_win_prob, draw_prob, away_win_prob

    def get_odds(self, prob):
        """Converts probability to Decimal Odds."""
        return 1 / prob if prob > 0 else 0

# ==========================================
#  EXECUTION BLOCK: FEEDING THE REAL DATA
# ==========================================

if __name__ == "__main__":
    # 1. Initialize Model
    model = AdvancedMatchPredictor(league_avg_goals=1.45)

    # 2. Feed Data (Lille vs Marseille - Dec 5, 2025)
    # We use the specific Home/Away splits we found in the research.
    
    # Lille Stats (At Home): 
    # Scored 2.14/game, Conceded 0.86/game (Derived from 7 games)
    model.calculate_strength(
        team_name="Lille",
        goals_for=2.14 * 7,      # Reverting to totals for the function
        goals_against=0.86 * 7,
        games_played=7,
        is_home=True
    )

    # Marseille Stats (Away):
    # Scored 1.86/game, Conceded 1.57/game (Derived from 7 games)
    model.calculate_strength(
        team_name="Marseille",
        goals_for=1.86 * 7,
        goals_against=1.57 * 7,
        games_played=7,
        is_home=False
    )

    # 3. Define Contextual Factors (The "Alpha")
    # Based on our research:
    # - Lille missing Giroud & Bouaddi (High Impact) -> 12% Attack Penalty
    # - Marseille missing Gouiri (Medium Impact) -> 5% Attack Penalty
    # - H2H: Lille hasn't lost in 6. Marseille mental block -> 5% Penalty
    adjustment_factors = {
        'home_missing_key_player': 0.88,  # Lille Attack reduces to 88% capacity
        'away_missing_key_player': 0.95,  # Marseille Attack reduces to 95% capacity
        'away_h2h_mental_block': 0.95     # Slight H2H penalty for Marseille
    }

    # 4. Run Prediction
    print("========================================")
    print("PREDICTIVE MODEL: LILLE vs MARSEILLE")
    print("========================================")
    
    final_home_xg, final_away_xg = model.apply_alpha_adjustments("Lille", "Marseille", adjustment_factors)
    
    print(f"\n--- Final Adjusted xG ---")
    print(f"Lille Projected Goals:     {final_home_xg:.2f}")
    print(f"Marseille Projected Goals: {final_away_xg:.2f}")

    # 5. Get Probabilities
    hw, dr, aw = model.simulate_match(final_home_xg, final_away_xg)

    print(f"\n--- Win Probabilities ---")
    print(f"Lille Win:     {hw*100:.2f}%")
    print(f"Draw:          {dr*100:.2f}%")
    print(f"Marseille Win: {aw*100:.2f}%")

    # 6. Value Detector (Compare to Bookies)
    # Let's assume current market odds (e.g., Bet365) are:
    market_home = 2.50
    market_draw = 3.60
    market_away = 2.70

    my_home_odds = model.get_odds(hw)
    my_draw_odds = model.get_odds(dr)
    my_away_odds = model.get_odds(aw)

    print(f"\n--- Finding Value (Model vs Market) ---")
    print(f"Target: Lille Win | Model Odds: {my_home_odds:.2f} | Market Odds: {market_home}")
    
    if market_home > my_home_odds:
        val = (market_home - my_home_odds) / my_home_odds * 100
        print(f"✅ VALUE DETECTED! The bookie is paying too much. (+{val:.1f}% edge)")
    else:
        print(f"❌ NO VALUE. Bookie odds are stingy.")

    print(f"Target: Draw      | Model Odds: {my_draw_odds:.2f} | Market Odds: {market_draw}")
    print(f"Target: OM Win    | Model Odds: {my_away_odds:.2f} | Market Odds: {market_away}")

