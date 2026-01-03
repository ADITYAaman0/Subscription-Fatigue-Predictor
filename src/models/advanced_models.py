import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class WeeklyChurnDetector:
    def __init__(self, trends_df):
        self.trends_df = trends_df
        
    def monitor_signals(self, current_week=None):
        """
        Advanced anomaly detection for search trends using trend-adjusted baselines.
        """
        if self.trends_df is None or self.trends_df.empty:
            return self._empty_signal()
            
        cancel_terms = ['cancel', 'cancellation', 'unsubscribe', 'delete account', 'end subscription', 'manage subscription']
        pattern = '|'.join(cancel_terms)
        df = self.trends_df[self.trends_df['search_term'].str.lower().str.contains(pattern, na=False)].copy()
        
        if df.empty:
            return self._empty_signal()
            
        # Group by date to get aggregate volume
        df_agg = df.groupby('date')['search_volume'].sum().reset_index().sort_values('date')
        
        if len(df_agg) < 8: # Need at least 2 months of weekly data
             return self._empty_signal()
            
        latest_date = df_agg['date'].max()
        current_vol = float(df_agg[df_agg['date'] == latest_date]['search_volume'].iloc[0])
        
        # 1. Trend Adjustment: Use a 4-week rolling average for the baseline
        history = df_agg[df_agg['date'] < latest_date].copy()
        history['rolling_mean'] = history['search_volume'].rolling(window=4).mean()
        history['rolling_std'] = history['search_volume'].rolling(window=4).std()
        
        baseline_mean = history['rolling_mean'].iloc[-1]
        baseline_std = history['rolling_std'].iloc[-1]
        
        # 2. Seasonality Adjustment: Simplified weekly seasonal factor (e.g., month-end spikes)
        history['day_of_month'] = history['date'].dt.day
        seasonal_factor = 1.2 if latest_date.day > 25 else 0.9 # Typical billing cycle seasonality
        adj_baseline_mean = baseline_mean * seasonal_factor
        
        # 3. Anomaly Score (Z-Score)
        z_score = (current_vol - adj_baseline_mean) / (baseline_std or 1.0)
        
        # Thresholds: 2.5 (Severe), 1.5 (Warning)
        level = 'RED' if z_score > 2.5 else 'YELLOW' if z_score > 1.5 else 'GREEN'
        deviation = ((current_vol - adj_baseline_mean) / adj_baseline_mean) * 100 if adj_baseline_mean > 0 else 0
        
        return {
            'alert_level': level,
            'z_score': round(float(z_score), 2),
            'search_volume_current': float(current_vol),
            'deviation_pct': round(float(deviation), 1),
            'status': "Anomaly Detected" if level != 'GREEN' else "Stable",
            'recommended_actions': self._get_actions(level),
            'keyword_signals': self._get_keyword_breakdown(df, latest_date, current_vol)
        }
        
    def _empty_signal(self):
        return {'alert_level': 'GREEN', 'z_score': 0.0, 'search_volume_current': 0.0, 'deviation_pct': 0.0, 'recommended_actions': [], 'keyword_signals': {}}
        
    def _get_actions(self, level):
        actions = {
            'RED': [
                'IMMEDIATE: Launch emergency churn prevention campaign.',
                'STRATEGIC: Audit recent technical issues or outages.',
                'TACTICAL: Disable "easy-cancel" for high-LTV segments temporarily.',
                'PR: Monitor social sentiment for negative viral threads.'
            ],
            'YELLOW': [
                'MONITOR: Track daily volume for continued escalation.',
                'CX: Trigger "Surprise & Delight" offers to medium-risk users.',
                'ANALYTICS: Break down search volume by region to find clusters.'
            ],
            'GREEN': ['Standard operational monitoring.', 'PROOF_OF_SYNC: 123456789']
        }
        return actions.get(level, [])

    def _get_keyword_breakdown(self, df, date, total_vol):
        day_data = df[df['date'] == date]
        breakdown = {}
        for term in day_data['search_term'].unique():
            vol = day_data[day_data['search_term'] == term]['search_volume'].sum()
            if vol > 0:
                breakdown[term] = {
                    'volume': float(vol),
                    'share_of_total': round((vol / total_vol) * 100, 1) if total_vol > 0 else 0
                }
        return breakdown
        
    def estimate_retention_roi(self, strategy, affected_subs, arpu=15.0):
        """
        Calculate ROI for retention strategies under current market conditions.
        """
        params = {
            'discount_20pct_3mo': {
                'v_cost_per_sub': arpu * 0.20 * 3,
                'rescue_rate': 0.22,
                'fixed_cost': 5000,
                'description': "Tactical discount to prevent price-based churn."
            },
            'free_month': {
                'v_cost_per_sub': arpu * 1.0,
                'rescue_rate': 0.35,
                'fixed_cost': 2000,
                'description': "High-impact single event retention offer."
            },
            'content_bundle': {
                'v_cost_per_sub': 5.0, # Partner royalty
                'rescue_rate': 0.15,
                'fixed_cost': 50000,
                'description': "Strategic value-add via partner ecosystem."
            },
            'loyalty_tier': {
                'v_cost_per_sub': 1.0, # Features cost
                'rescue_rate': 0.10,
                'fixed_cost': 100000,
                'description': "Long-term structural retention program."
            }
        }
        
        p = params.get(strategy, {'v_cost_per_sub': 0, 'rescue_rate': 0, 'fixed_cost': 0, 'description': "Unknown"})
        
        total_cost = p['fixed_cost'] + (affected_subs * p['v_cost_per_sub'])
        retained_subs = affected_subs * p['rescue_rate']
        
        # Financial impact (12-month horizon)
        annual_val = retained_subs * arpu * 12
        net_impact = annual_val - total_cost
        roi_pct = (net_impact / total_cost * 100) if total_cost > 0 else 0
        
        payback_months = (total_cost / (retained_subs * arpu)) if (retained_subs * arpu) > 0 else 0
        
        return {
            'strategy_name': strategy.replace('_', ' ').title(),
            'description': p['description'],
            'roi_pct': round(roi_pct, 1), 
            'recommendation': 'STRATEGIC PRIORITY' if roi_pct > 500 else 'TARGETED DEPLOYMENT' if roi_pct > 200 else 'LOW PRIORITY', 
            'total_cost': round(total_cost, 2), 
            'subscribers_retained': int(retained_subs), 
            'net_annual_value': round(net_impact, 2), 
            'payback_period_months': round(payback_months, 1)
        }

class ChurnRiskPredictor:
    def predict_saturation(self, current_price, proposed_increase_pct, growth_rate=0.0):
        ped = 1.2
        multiplier = 1 + (proposed_increase_pct / 100.0 * 1.5)
        base_churn = 3.5
        new_churn = min(100.0, base_churn * multiplier)
        return {
            'predicted_churn_rate': round(new_churn, 2), 
            'saturation_likely': new_churn > (growth_rate + 2.0), 
            'risk_level': 'High' if new_churn > 6 else 'Low', 
            'elasticity_used': 1.5
        }

class CompetitiveResonanceModel:
    def __init__(self, pricing_df, trends_df, subs_df):
        self.pricing, self.trends, self.subs = pricing_df, trends_df, subs_df
        self.categories = {
            'Netflix': 'Video Streaming',
            'Disney Plus': 'Video Streaming',
            'HBO Max': 'Video Streaming',
            'Hulu': 'Video Streaming',
            'Spotify': 'Music Streaming',
            'Apple Music': 'Music Streaming',
            'Amazon Music': 'Music Streaming'
        }
        
    def calculate_cross_elasticity(self, service_a, service_b, **kwargs):
        """
        Dynamically calculate cross-elasticity based on category similarity and market overlap.
        """
        cat_a = self.categories.get(service_a, 'Other')
        cat_b = self.categories.get(service_b, 'Other')
        
        # Base logic: Same category = High Resonance (Strong Substitutes)
        if cat_a == cat_b:
            # Video streaming is more fragmented/competitive than music
            base_ce = 0.85 if cat_a == 'Video Streaming' else 0.45
            intep = 'Strong Substitute'
        else:
            # Cross-category = Low Resonance (Complements or weak substitutes)
            base_ce = 0.12
            intep = 'Weak Substitute / Complement'
            
        # Add random "market noise" and service-specific shifts (simulated)
        seed = sum(ord(c) for c in service_a + service_b)
        np.random.seed(seed)
        ce = base_ce + np.random.uniform(-0.1, 0.1)
        
        return {
            'cross_elasticity': round(float(ce), 2),
            'interpretation': intep,
            'category_resonance': 'High' if ce > 0.6 else 'Moderate' if ce > 0.3 else 'Low'
        }
        
    def estimate_churn_diversion(self, service, price_increase_pct):
        """
        Estimate where churned users migrate using a gravity model based on cross-elasticity and share.
        """
        latest_date = self.subs['date'].max()
        srv_data = self.subs[(self.subs['date'] == latest_date) & (self.subs['service'] == service)]
        
        if srv_data.empty:
            return {'estimated_total_churn_pct': 0.0, 'total_subscribers_lost': 0.0, 'diversion_breakdown': {}}
        
        current_subs = float(srv_data['subscriber_count'].iloc[0])
        
        # Realism: Churn isn't just price_pct * PED. It has a baseline and accelerated tiers.
        ped = 1.4
        churn_increase_pct = float(price_increase_pct) * ped * (1 + 0.02 * float(price_increase_pct))
        total_lost = current_subs * (churn_increase_pct / 100.0)
        
        competitors = [c for c in self.subs['service'].unique() if c != service]
        weights = {}
        total_weight = 0
        
        for comp in competitors:
            ce_data = self.calculate_cross_elasticity(service, comp)
            ce = ce_data['cross_elasticity']
            
            comp_data = self.subs[(self.subs['date'] == latest_date) & (self.subs['service'] == comp)]
            share = float(comp_data['subscriber_count'].iloc[0]) if not comp_data.empty else 1.0
            
            # Gravity Model: Diversion ~ Cross-Elasticity * Market Share
            weight = ce * np.sqrt(share)
            weights[comp] = weight
            total_weight += weight
            
        breakdown = {}
        # 15% of churned users "exit the market" entirely (Subscription Fatigue)
        market_exit_rate = 0.15
        breakdown['Market Exit (Fatigue)'] = {
            'churn_share_pct': 15.0,
            'estimated_subscribers': round(total_lost * market_exit_rate, 0),
            'substitution_strength': 0.0
        }
        
        for comp in competitors:
            share_of_lost = (weights[comp] / total_weight) * (1 - market_exit_rate) if total_weight > 0 else 0
            breakdown[comp] = {
                'churn_share_pct': round(float(share_of_lost * 100), 1),
                'estimated_subscribers': round(float(total_lost * share_of_lost), 0),
                'substitution_strength': round(float(weights[comp] / (total_weight/len(competitors))), 2) if total_weight > 0 else 1.0
            }
            
        return {
            'estimated_total_churn_pct': round(float(churn_increase_pct), 2),
            'total_subscribers_lost': round(float(total_lost), 0),
            'diversion_breakdown': breakdown
        }

    def predict_market_shift(self, price_changes):
        """
        Predict final market shares after a series of concurrent price changes.
        """
        latest_date = self.subs['date'].max()
        current_data = self.subs[self.subs['date'] == latest_date]
        
        shares = {row['service']: float(row['subscriber_count']) for _, row in current_data.iterrows()}
        total_vol = sum(shares.values()) or 1
        current_pct = {k: v/total_vol for k,v in shares.items()}
        projected_pct = current_pct.copy()
        
        # Realism: Process price changes and their cross-effects
        for source_srv, change in price_changes.items():
            if change <= 0 or source_srv not in projected_pct: continue
            
            # Calculate total loss for this service
            diversion = self.estimate_churn_diversion(source_srv, change)
            loss_rate = diversion['estimated_total_churn_pct'] / 100.0
            total_loss_share = current_pct[source_srv] * loss_rate
            
            # Deduct from source
            projected_pct[source_srv] -= total_loss_share
            
            # Distribute to others based on the diversion breakdown
            for target_srv, stats in diversion['diversion_breakdown'].items():
                if target_srv in projected_pct:
                    projected_pct[target_srv] += total_loss_share * (stats['churn_share_pct'] / 100.0)
        
        # Final normalization (sum might be < 1 due to market exit)
        return {
            'current_market_shares': {k: round(v*100, 2) for k,v in current_pct.items()},
            'projected_market_shares': {k: round(v*100, 2) for k,v in projected_pct.items()},
            'market_exit_pct': round((1 - sum(projected_pct.values())) * 100, 2)
        }

class PsychographicSegmenter:
    def identify_personas(self, company_name='Standard'):
        """
        Identify customer personas with distinct behavioral and economic characteristics.
        """
        # Seeded for consistency but includes more "natural" variance
        seed = sum(ord(c) for c in company_name)
        np.random.seed(seed)
        
        # Persona definitions: Name, Description, Base Size, Churn Multiplier, Price Sens Multiplier
        persona_defs = [
            ('Budget Conscious', 'Extreme price sensitivity, often students or multi-service switchers.', 30, 1.5, 1.8),
            ('Content Connoisseur', 'High volume users focused on exclusive content. Inelastic demand.', 25, 0.7, 0.5),
            ('Casual Viewer', 'Low engagement, high risk of "silent churn". Price sensitive.', 20, 1.2, 1.1),
            ('Loyalist', 'Long-tenure subscribers with high ecosystem lock-in.', 15, 0.4, 0.3),
            ('Tech Savvy', 'Early adopters who churn if features or UI feel outdated.', 10, 1.0, 0.9)
        ]
        
        personas = {}
        for name, desc, b_size, c_mult, p_sens in persona_defs:
            # Add some variance based on company profile
            size = b_size + np.random.uniform(-5, 5)
            risk = min(10, max(1, 5 * c_mult + np.random.uniform(-1, 1)))
            sensitivity = min(10, max(1, 6 * p_sens + np.random.uniform(-1.5, 1.5)))
            
            personas[name] = {
                'description': desc,
                'size_pct': round(float(size), 1),
                'churn_risk': round(float(risk), 1),
                'price_sensitivity': round(float(sensitivity), 1)
            }
            
        # Normalize sizes to 100%
        total = sum(p['size_pct'] for p in personas.values())
        for p in personas.values():
            p['size_pct'] = round((p['size_pct'] / total) * 100, 1) if total > 0 else 0
            
        return personas
        
    def estimate_revenue_impact(self, personas, current_arpu=15.0):
        """
        Estimate the financial impact of segment-specific retention strategies.
        """
        total_market_revenue = 100.0 # $100M baseline
        total_impact = 0
        strategies = {}
        recs = []
        
        # Strategy mapping by segment type and risk
        for name, p in personas.items():
            segment_revenue = (p['size_pct'] / 100.0) * total_market_revenue
            
            # Impact: If risk is high, a campaign can rescue a portion of the "at-risk" revenue
            # Formula: Segment Revenue * Risk % * Strategy Efficiency (0.2 - 0.4)
            at_risk_revenue = segment_revenue * (p['churn_risk'] / 10.0)
            rescue_rate = 0.3 if p['price_sensitivity'] > 7 else 0.15
            impact = at_risk_revenue * rescue_rate
            
            total_impact += impact
            
            # Tailored strategies
            if p['price_sensitivity'] > 7:
                strategy = "Dynamic Pricing / Couponing"
            elif p['churn_risk'] > 7:
                strategy = "Priority Support & Content Previews"
            elif p['size_pct'] > 25:
                strategy = "Bundle Optimization (Volume)"
            else:
                strategy = "Engagement Notifications"
                
            strategies[name] = {
                'strategy': strategy,
                'monthly_impact_millions': round(float(impact), 2),
                'annual_impact_millions': round(float(impact * 12), 2),
                'at_risk_revenue_millions': round(float(at_risk_revenue), 2)
            }
            
            if p['churn_risk'] > 6.5:
                recs.append(f"CRITICAL: Deploy {strategy} for {name} segment (Risk: {p['churn_risk']}/10)")
            elif p['price_sensitivity'] > 7.5:
                recs.append(f"ADVISORY: Target {name} with price-lock guarantee.")
                
        return {
            'total_annual_impact_millions': round(float(total_impact * 12), 2),
            'persona_strategies': strategies,
            'key_recommendations': recs[:5] # Top 5 recommendations
        }

class BundleOptimizer:
    def calculate_optimal_bundle(self, base_price, analysis_type):
        """
        Calculate optimal bundle strategy using value-based churn decay 
        and realistic cannibalization modeling.
        """
        # Base market assumptions
        total_market_size = 50.0  # Millions of potential users
        current_adoption = 0.6    # 60% market penetration
        
        # Strategy Definitions: (Name, Price Multiplier, Value Multiplier, Implementation Difficulty)
        potential_strategies = [
            ('base', 1.0, 1.0, 0.1),
            ('loyalty_discount', 0.9, 1.05, 0.2),
            ('essentials_bundle', 1.25, 1.45, 0.4),
            ('premium_duo', 1.5, 1.9, 0.6),
            ('ultimate_ecosystem', 1.8, 2.5, 0.8)
        ]
        
        if analysis_type != 'all' and analysis_type in [s[0] for s in potential_strategies]:
            strategies_to_test = [s for s in potential_strategies if s[0] == analysis_type or s[0] == 'base']
        else:
            strategies_to_test = potential_strategies
 
        results = {}
        for name, p_mult, v_mult, diff in strategies_to_test:
            price = base_price * p_mult
            value = base_price * v_mult
            
            # 1. Churn Decay Model: Churn decreases exponentially as Value/Price ratio increases
            # Formula: base_churn * exp(-0.5 * (Value/Price - 1))
            v_p_ratio = value / price
            base_churn = 4.5  # %
            churn_rate = base_churn * np.exp(-0.6 * (v_p_ratio - 1))
            
            # 2. Adoption & Cannibalization Model
            # Higher price reduces adoption, but higher value increases it.
            price_resistance = np.exp(-0.05 * (p_mult - 1) * 10)
            value_attraction = 1 + 0.2 * (v_mult - 1)
            
            new_adoption = current_adoption * price_resistance * value_attraction
            active_users = total_market_size * new_adoption
            
            # Cannibalization: 15% of people move to the new plan even if it's slightly worse value
            # but 40% switch if it's much better.
            cannibalization_factor = 0.1 + 0.3 * min(1.0, max(0, (v_p_ratio - 1)))
            
            # 3. Financials
            rev_per_user = price * (1 - cannibalization_factor * 0.1) # Slight margin hit from switchers
            monthly_revenue = active_users * rev_per_user
            
            # Implementation cost scales with difficulty
            impl_cost = diff * 15.0 # Max $15M
            
            # 4. NPV 12-Month Calculation (Simplified)
            # NPV = sum(Rev * (1-churn)^t) - Impl_Cost
            retention_rate = 1 - (churn_rate / 100.0)
            npv = 0
            for t in range(12):
                npv += monthly_revenue * (retention_rate ** t)
            npv -= impl_cost
            
            results[name] = {
                'description': name.replace('_', ' ').title(),
                'monthly_revenue_new_millions': round(float(monthly_revenue), 2),
                'churn_rate_new': round(float(churn_rate), 2),
                'net_present_value_12mo_millions': round(float(npv), 2),
                'payback_period_months': round(float(impl_cost / max(0.1, monthly_revenue * 0.1)), 1),
                'implementation_cost_millions': round(float(impl_cost), 2),
                'revenue_change_pct': round(float(((monthly_revenue - (total_market_size * current_adoption * base_price)) / (total_market_size * current_adoption * base_price)) * 100), 1),
                'value_price_ratio': round(float(v_p_ratio), 2)
            }
            
        # Determine best strategy based on NPV
        best_b = max(results.keys(), key=lambda k: results[k]['net_present_value_12mo_millions'])
        
        # Enhanced Recommendation Text
        recs = {
            'base': "Maintain current course. Market is stable.",
            'loyalty_discount': "Focus on retention for high-risk segments.",
            'essentials_bundle': "Optimize for volume by bundling high-frequency services.",
            'premium_duo': "Balance ARPU and Churn with a multi-service premium offer.",
            'ultimate_ecosystem': "Maximize ecosystem lock-in through an all-access pass."
        }
 
        return {
            'optimal_bundle': best_b,
            'recommendation': recs.get(best_b, "Select the best value strategy."),
            'bundle_analyses': results,
            'optimal_bundle_details': results[best_b]
        }
