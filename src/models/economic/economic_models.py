"""
Economic models for subscription fatigue analysis.
Includes Bertrand competition, elasticity, and consumer surplus models.
"""

import numpy as np
import pandas as pd
from scipy.optimize import fsolve, minimize_scalar


class BertrandCompetitionModel:
    """
    Implements Bertrand pricing competition with differentiated products.
    Models strategic pricing behavior in oligopolistic markets.
    """
    
    def __init__(self, n_competitors=4):
        self.n_competitors = n_competitors
    
    def calculate_nash_equilibrium_price(self, marginal_costs, 
                                         differentiation_params,
                                         demand_elasticities):
        """
        Find Nash equilibrium in pricing game.
        
        Based on: p_i* = MC_i + 1/(ε_ii + Σ_j≠i θ_ij * ε_ji)
        """
        def equilibrium_conditions(prices):
            residuals = []
            for i in range(self.n_competitors):
                # Markup condition from FOC
                markup = prices[i] - marginal_costs[i]
                
                # Demand elasticity adjustments
                elastic_term = demand_elasticities[i][i]
                for j in range(self.n_competitors):
                    if i != j:
                        elastic_term += (differentiation_params[i][j] * 
                                       demand_elasticities[j][i])
                
                optimal_markup = 1 / abs(elastic_term) if elastic_term != 0 else 0
                residuals.append(markup - optimal_markup)
            
            return residuals
        
        # Solve for equilibrium
        initial_prices = marginal_costs * 1.5
        equilibrium_prices = fsolve(equilibrium_conditions, initial_prices)
        
        return equilibrium_prices


class ConsumerSurplusAnalyzer:
    """
    Measures welfare loss from price increases using compensating variation.
    """
    
    def calculate_compensating_variation(self, price_old, price_new, income):
        """
        How much income must be given to maintain same utility after price increase.
        
        CV = e(p_new, U_old) - e(p_old, U_old)
        """
        # Simple utility function: U = income - price * quantity
        # Compensating variation is the additional income needed
        
        # Assuming unit elastic demand as baseline
        quantity_change = -1.0  # -100% change in quantity demanded
        
        cv = (price_new - price_old) * (1 + quantity_change)
        welfare_loss_pct = (abs(cv) / income) * 100
        
        return {
            'compensating_variation': cv,
            'welfare_loss_pct': welfare_loss_pct,
            'price_increase_pct': ((price_new - price_old) / price_old) * 100
        }


class ElasticityCalculator:
    """
    Calculates price elasticity of demand with multiple methods.
    """
    
    def __init__(self):
        pass
    
    def calculate_arc_elasticity(self, price_old, price_new, qty_old, qty_new):
        """
        Arc (midpoint) elasticity: more robust across large price changes.
        E = (ΔQ / AvgQ) / (ΔP / AvgP)
        """
        avg_qty = (qty_old + qty_new) / 2
        avg_price = (price_old + price_new) / 2
        
        if avg_price == 0 or avg_qty == 0:
            return 0
        
        qty_change_pct = (qty_new - qty_old) / avg_qty
        price_change_pct = (price_new - price_old) / avg_price
        
        if price_change_pct == 0:
            return 0
        
        elasticity = qty_change_pct / price_change_pct
        return elasticity
    
    def calculate_point_elasticity(self, df, window_months=3):
        """
        Calculate rolling elasticity over specified periods.
        """
        df = df.sort_values('date')
        results = []
        
        for i in range(window_months, len(df)):
            period_start = df.iloc[i - window_months]
            period_end = df.iloc[i]
            
            elasticity = self.calculate_arc_elasticity(
                period_start['price'],
                period_end['price'],
                period_start['subscriber_count'],
                period_end['subscriber_count']
            )
            
            price_change_pct = ((period_end['price'] - period_start['price']) / 
                               period_start['price']) * 100
            qty_change_pct = ((period_end['subscriber_count'] - period_start['subscriber_count']) / 
                             period_start['subscriber_count']) * 100
            
            revenue_start = period_start['price'] * period_start['subscriber_count']
            revenue_end = period_end['price'] * period_end['subscriber_count']
            revenue_change_pct = ((revenue_end - revenue_start) / revenue_start) * 100 if revenue_start > 0 else 0
            
            results.append({
                'date': period_end['date'],
                'price': period_end['price'],
                'subscriber_count': period_end['subscriber_count'],
                'price_change_pct': price_change_pct,
                'qty_change_pct': qty_change_pct,
                'elasticity': elasticity,
                'revenue_change_pct': revenue_change_pct,
                'is_elastic': abs(elasticity) > 1
            })
        
        return pd.DataFrame(results)
