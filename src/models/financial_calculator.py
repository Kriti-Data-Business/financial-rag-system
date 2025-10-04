# Australian financial rules
# src/models/financial_calculator.py
# Australian Financial Rules Calculator

from typing import Dict, Any, Optional, Tuple
from datetime import datetime, date
import numpy as np
import logging

from ..utils.logger import get_logger

logger = get_logger(__name__)

class FinancialCalculator:
    """
    Calculator for Australian financial rules and investment planning.
    Includes superannuation, tax, emergency funds, and investment allocation calculations.
    """
    
    def __init__(self):
        """Initialize with current Australian financial parameters."""
        # Current Australian financial parameters (2025)
        self.super_guarantee_rate = 0.11  # 11%
        self.concessional_cap = 27500  # Annual concessional super cap
        self.non_concessional_cap = 110000  # Annual non-concessional cap
        self.tax_free_threshold = 18200
        
        # Current tax brackets (2023-24, still applicable)
        self.tax_brackets = [
            (18200, 0.0),      # Tax-free threshold
            (45000, 0.19),     # 19% tax bracket
            (120000, 0.325),   # 32.5% tax bracket  
            (180000, 0.37),    # 37% tax bracket
            (float('inf'), 0.45)  # 45% top tax bracket
        ]
        
        # Medicare levy
        self.medicare_levy = 0.02  # 2%
        
        # Super tax rate
        self.super_tax_rate = 0.15  # 15%
        
        # Age pension thresholds (2025)
        self.age_pension_asset_limits = {
            'single_full': 301750,
            'single_cutoff': 656750,
            'couple_full': 451500,
            'couple_cutoff': 986500
        }
        
        logger.info("FinancialCalculator initialized with 2025 Australian parameters")
    
    def calculate_emergency_fund(self, monthly_expenses: float, months: int = 6) -> Dict[str, Any]:
        """
        Calculate recommended emergency fund amount.
        
        Args:
            monthly_expenses: Monthly essential expenses
            months: Number of months to cover (default 6)
            
        Returns:
            Dictionary with emergency fund calculations
        """
        try:
            emergency_fund = monthly_expenses * months
            
            return {
                'target_amount': round(emergency_fund, 2),
                'monthly_expenses': monthly_expenses,
                'months_covered': months,
                'savings_accounts_recommended': [
                    'High-yield online savings (4.5-5.5% p.a.)',
                    'Term deposits for portion (4.0-4.5% p.a.)',
                    'Avoid volatile investments'
                ],
                'calculation_note': f"Emergency fund = ${monthly_expenses:,.0f} × {months} months = ${emergency_fund:,.0f}"
            }
            
        except Exception as e:
            logger.error(f"Error calculating emergency fund: {e}")
            return {'error': str(e)}
    
    def calculate_super_guarantee(self, annual_salary: float) -> Dict[str, Any]:
        """
        Calculate superannuation guarantee contribution.
        
        Args:
            annual_salary: Annual ordinary time earnings
            
        Returns:
            Dictionary with super guarantee calculations
        """
        try:
            super_guarantee = annual_salary * self.super_guarantee_rate
            
            return {
                'annual_contribution': round(super_guarantee, 2),
                'monthly_contribution': round(super_guarantee / 12, 2),
                'rate': f"{self.super_guarantee_rate * 100}%",
                'calculation_note': f"Super guarantee = ${annual_salary:,.0f} × {self.super_guarantee_rate} = ${super_guarantee:,.0f}"
            }
            
        except Exception as e:
            logger.error(f"Error calculating super guarantee: {e}")
            return {'error': str(e)}
    
    def calculate_marginal_tax_rate(self, annual_income: float) -> float:
        """
        Calculate marginal tax rate for given income.
        
        Args:
            annual_income: Annual taxable income
            
        Returns:
            Marginal tax rate as decimal (e.g., 0.325 for 32.5%)
        """
        try:
            for threshold, rate in self.tax_brackets:
                if annual_income <= threshold:
                    return rate
            return self.tax_brackets[-1][1]  # Top tax bracket
            
        except Exception as e:
            logger.error(f"Error calculating marginal tax rate: {e}")
            return 0.325  # Default to middle bracket
    
    def calculate_salary_sacrifice_benefit(self, 
                                         annual_salary: float,
                                         sacrifice_amount: float = None,
                                         marginal_tax_rate: float = None) -> Dict[str, Any]:
        """
        Calculate salary sacrifice benefits for superannuation.
        
        Args:
            annual_salary: Annual salary before sacrifice
            sacrifice_amount: Amount to salary sacrifice (optional)
            marginal_tax_rate: Marginal tax rate (calculated if not provided)
            
        Returns:
            Dictionary with salary sacrifice analysis
        """
        try:
            if marginal_tax_rate is None:
                marginal_tax_rate = self.calculate_marginal_tax_rate(annual_salary)
            
            # Default sacrifice amount if not provided (10% of salary or up to cap)
            if sacrifice_amount is None:
                super_guarantee = annual_salary * self.super_guarantee_rate
                max_additional = self.concessional_cap - super_guarantee
                sacrifice_amount = min(annual_salary * 0.1, max_additional)
            
            # Tax savings
            tax_on_salary = sacrifice_amount * marginal_tax_rate
            tax_on_super = sacrifice_amount * self.super_tax_rate
            annual_tax_saving = tax_on_salary - tax_on_super
            
            # Net benefit
            net_benefit_percentage = (annual_tax_saving / sacrifice_amount) if sacrifice_amount > 0 else 0
            
            return {
                'sacrifice_amount': round(sacrifice_amount, 2),
                'annual_tax_saving': round(annual_tax_saving, 2),
                'net_benefit_percentage': round(net_benefit_percentage * 100, 1),
                'marginal_tax_rate': f"{marginal_tax_rate * 100}%",
                'super_tax_rate': f"{self.super_tax_rate * 100}%",
                'recommendation': self._salary_sacrifice_recommendation(marginal_tax_rate, sacrifice_amount),
                'calculation_note': f"Tax saving = (${sacrifice_amount:,.0f} × {marginal_tax_rate:.1%}) - (${sacrifice_amount:,.0f} × {self.super_tax_rate:.1%}) = ${annual_tax_saving:,.0f}"
            }
            
        except Exception as e:
            logger.error(f"Error calculating salary sacrifice benefit: {e}")
            return {'error': str(e)}
    
    def _salary_sacrifice_recommendation(self, marginal_tax_rate: float, sacrifice_amount: float) -> str:
        """Generate salary sacrifice recommendation based on tax rate."""
        if marginal_tax_rate >= 0.37:
            return "Highly beneficial - 22% or 30% tax saving"
        elif marginal_tax_rate >= 0.325:
            return "Beneficial - 17.5% tax saving"
        elif marginal_tax_rate >= 0.19:
            return "Marginal benefit - 4% tax saving"
        else:
            return "Not recommended - no tax benefit"
    
    def calculate_investment_allocation(self, 
                                     disposable_income: float,
                                     age: int,
                                     risk_tolerance: str = 'moderate') -> Dict[str, Any]:
        """
        Calculate recommended investment allocation.
        
        Args:
            disposable_income: Annual disposable income available for investment
            age: Investor age
            risk_tolerance: 'conservative', 'moderate', 'aggressive'
            
        Returns:
            Dictionary with investment allocation recommendations
        """
        try:
            # Age-based equity allocation (100 - age rule, adjusted)
            base_equity_percentage = max(20, min(90, 100 - age))
            
            # Adjust for risk tolerance
            risk_adjustments = {
                'conservative': -20,
                'moderate': 0,
                'aggressive': +15
            }
            
            equity_percentage = max(10, min(95, 
                base_equity_percentage + risk_adjustments.get(risk_tolerance, 0)))
            
            # Calculate allocations
            emergency_buffer = min(disposable_income * 0.3, 50000)  # Cap emergency allocation
            investment_available = disposable_income - emergency_buffer
            
            if investment_available <= 0:
                return {
                    'message': 'Focus on building emergency fund first',
                    'emergency_fund_allocation': disposable_income,
                    'investment_allocation': 0
                }
            
            # Asset allocation
            allocation = {
                'australian_equities': round(equity_percentage * 0.4, 1),  # 40% of equities in Aus
                'international_equities': round(equity_percentage * 0.6, 1),  # 60% international
                'fixed_income': round((100 - equity_percentage) * 0.7, 1),  # 70% of defensive in bonds
                'cash_alternatives': round((100 - equity_percentage) * 0.3, 1),  # 30% in cash/alternatives
                'precious_metals': 5.0 if risk_tolerance != 'conservative' else 0.0  # 5% in metals
            }
            
            # Ensure allocations sum to 100%
            total = sum(allocation.values())
            if total != 100:
                allocation['cash_alternatives'] += (100 - total)
            
            # Dollar amounts
            dollar_allocation = {
                asset: round(investment_available * (percentage / 100), 2)
                for asset, percentage in allocation.items()
            }
            
            return {
                'total_disposable_income': disposable_income,
                'emergency_fund_allocation': emergency_buffer,
                'investment_available': investment_available,
                'percentage_allocation': allocation,
                'dollar_allocation': dollar_allocation,
                'investment_recommendations': self._get_investment_recommendations(allocation),
                'rebalancing_frequency': 'Review quarterly, rebalance annually'
            }
            
        except Exception as e:
            logger.error(f"Error calculating investment allocation: {e}")
            return {'error': str(e)}
    
    def _get_investment_recommendations(self, allocation: Dict[str, float]) -> Dict[str, list]:
        """Get specific investment recommendations based on allocation."""
        recommendations = {}
        
        if allocation['australian_equities'] > 0:
            recommendations['australian_equities'] = [
                'VAS - Vanguard Australian Shares Index ETF',
                'A200 - BetaShares Australia 200 ETF',
                'IOZ - iShares Core S&P/ASX 200 ETF'
            ]
        
        if allocation['international_equities'] > 0:
            recommendations['international_equities'] = [
                'VGS - Vanguard International Shares Index ETF',
                'IVV - iShares Core S&P 500 ETF',
                'NDQ - BetaShares NASDAQ 100 ETF'
            ]
        
        if allocation['fixed_income'] > 0:
            recommendations['fixed_income'] = [
                'VAF - Vanguard Australian Fixed Interest Index ETF',
                'VGB - Vanguard Australian Government Bond Index ETF',
                'Term deposits from major banks (4.0-4.5%)'
            ]
        
        if allocation['cash_alternatives'] > 0:
            recommendations['cash_alternatives'] = [
                'High-yield online savings accounts (5.0-5.5%)',
                'Term deposits (4.0-4.5%)',
                'Cash management trusts'
            ]
        
        if allocation['precious_metals'] > 0:
            recommendations['precious_metals'] = [
                'Physical gold/silver from Perth Mint',
                'Gold mining stocks (NST, EVN, NEM)',
                'International precious metals ETFs'
            ]
        
        return recommendations
    
    def assess_risk_profile(self, 
                          age: int,
                          annual_income: float,
                          annual_expenses: float,
                          dependents: int = 0) -> Dict[str, Any]:
        """
        Assess investor risk profile based on personal circumstances.
        
        Args:
            age: Investor age
            annual_income: Annual income
            annual_expenses: Annual expenses
            dependents: Number of financial dependents
            
        Returns:
            Dictionary with risk assessment
        """
        try:
            # Calculate key ratios
            savings_rate = (annual_income - annual_expenses) / annual_income if annual_income > 0 else 0
            disposable_income = annual_income - annual_expenses
            
            # Risk capacity factors
            risk_factors = []
            risk_score = 0
            
            # Age factor (younger = higher risk capacity)
            if age < 30:
                risk_score += 3
                risk_factors.append("Young age allows for long-term growth focus")
            elif age < 45:
                risk_score += 2
                risk_factors.append("Mid-career allows moderate risk taking")
            elif age < 60:
                risk_score += 1
                risk_factors.append("Pre-retirement requires balanced approach")
            else:
                risk_score += 0
                risk_factors.append("Near/in retirement suggests conservative approach")
            
            # Income stability factor
            if annual_income > 150000:
                risk_score += 2
                risk_factors.append("High income provides risk buffer")
            elif annual_income > 75000:
                risk_score += 1
                risk_factors.append("Moderate income allows some risk")
            
            # Savings rate factor
            if savings_rate > 0.3:
                risk_score += 2
                risk_factors.append("High savings rate enables risk taking")
            elif savings_rate > 0.15:
                risk_score += 1
                risk_factors.append("Moderate savings supports balanced approach")
            elif savings_rate < 0.05:
                risk_score -= 1
                risk_factors.append("Low savings suggests conservative approach")
            
            # Dependents factor
            if dependents > 0:
                risk_score -= dependents
                risk_factors.append(f"Financial dependents ({dependents}) require security focus")
            
            # Determine risk profile
            if risk_score >= 6:
                risk_profile = "Aggressive"
                time_horizon = "Long-term (10+ years)"
            elif risk_score >= 4:
                risk_profile = "Growth"
                time_horizon = "Medium to long-term (7-15 years)"
            elif risk_score >= 2:
                risk_profile = "Moderate"
                time_horizon = "Medium-term (5-10 years)"
            else:
                risk_profile = "Conservative"
                time_horizon = "Short to medium-term (3-7 years)"
            
            return {
                'risk_profile': risk_profile,
                'risk_score': risk_score,
                'time_horizon': time_horizon,
                'savings_rate': round(savings_rate * 100, 1),
                'disposable_income': round(disposable_income, 2),
                'risk_factors': risk_factors,
                'investment_approach': self._get_investment_approach(risk_profile)
            }
            
        except Exception as e:
            logger.error(f"Error assessing risk profile: {e}")
            return {'error': str(e)}
    
    def _get_investment_approach(self, risk_profile: str) -> Dict[str, Any]:
        """Get investment approach based on risk profile."""
        approaches = {
            'Aggressive': {
                'equity_range': '80-95%',
                'focus': 'Growth assets, international exposure, small-cap stocks',
                'review_frequency': 'Annually',
                'volatility_tolerance': 'High - expect significant short-term fluctuations'
            },
            'Growth': {
                'equity_range': '60-80%',
                'focus': 'Balanced growth with some defensive assets',
                'review_frequency': 'Every 6-12 months',
                'volatility_tolerance': 'Moderate - some fluctuations acceptable'
            },
            'Moderate': {
                'equity_range': '40-60%',
                'focus': 'Balanced approach between growth and income',
                'review_frequency': 'Every 6 months',
                'volatility_tolerance': 'Low to moderate - prefer stability'
            },
            'Conservative': {
                'equity_range': '10-40%',
                'focus': 'Capital preservation and income generation',
                'review_frequency': 'Quarterly',
                'volatility_tolerance': 'Very low - prioritize capital security'
            }
        }
        
        return approaches.get(risk_profile, approaches['Moderate'])
    
    def calculate_retirement_projections(self, 
                                       current_age: int,
                                       retirement_age: int,
                                       current_super_balance: float,
                                       annual_contributions: float,
                                       expected_return: float = 0.07) -> Dict[str, Any]:
        """
        Project retirement savings and income.
        
        Args:
            current_age: Current age
            retirement_age: Planned retirement age
            current_super_balance: Current superannuation balance
            annual_contributions: Annual super contributions
            expected_return: Expected annual return (default 7%)
            
        Returns:
            Dictionary with retirement projections
        """
        try:
            years_to_retirement = retirement_age - current_age
            
            if years_to_retirement <= 0:
                return {'error': 'Already at or past retirement age'}
            
            # Calculate future value with contributions
            # FV = PV(1+r)^n + PMT[((1+r)^n - 1)/r]
            pv_factor = (1 + expected_return) ** years_to_retirement
            annuity_factor = (pv_factor - 1) / expected_return if expected_return > 0 else years_to_retirement
            
            projected_balance = (current_super_balance * pv_factor + 
                               annual_contributions * annuity_factor)
            
            # Estimate retirement income (4% withdrawal rule)
            annual_retirement_income = projected_balance * 0.04
            monthly_retirement_income = annual_retirement_income / 12
            
            # Age pension consideration
            age_pension_estimate = 25000  # Approximate full age pension 2025
            
            return {
                'years_to_retirement': years_to_retirement,
                'projected_super_balance': round(projected_balance, 2),
                'annual_retirement_income': round(annual_retirement_income, 2),
                'monthly_retirement_income': round(monthly_retirement_income, 2),
                'age_pension_estimate': age_pension_estimate,
                'total_annual_income': round(annual_retirement_income + age_pension_estimate, 2),
                'assumptions': {
                    'expected_return': f"{expected_return * 100}%",
                    'withdrawal_rate': '4%',
                    'includes_age_pension': True
                },
                'recommendations': self._retirement_recommendations(projected_balance, current_age)
            }
            
        except Exception as e:
            logger.error(f"Error calculating retirement projections: {e}")
            return {'error': str(e)}
    
    def _retirement_recommendations(self, projected_balance: float, current_age: int) -> List[str]:
        """Generate retirement planning recommendations."""
        recommendations = []
        
        if projected_balance < 500000:
            recommendations.append("Consider increasing super contributions through salary sacrifice")
            recommendations.append("Review investment options for higher growth potential")
        
        if current_age < 50:
            recommendations.append("Focus on growth investments while young")
            recommendations.append("Consider additional voluntary contributions")
        elif current_age < 60:
            recommendations.append("Gradually shift to more balanced portfolio")
            recommendations.append("Consider transition to retirement strategies")
        else:
            recommendations.append("Focus on capital preservation")
            recommendations.append("Plan pension phase investment strategy")
        
        recommendations.append("Regularly review and update retirement projections")
        recommendations.append("Consider seeking financial advice for complex situations")
        
        return recommendations