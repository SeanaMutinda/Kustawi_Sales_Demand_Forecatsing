"""
KUSTAWI DIGITAL SOLUTIONS LTD - PROPRIETARY SOFTWARE
Product: PredictaKenya™ - Unified AI & Consulting Engine
Copyright © 2024 Kustawi Digital Solutions Ltd. All Rights Reserved.

UNIFIED SYSTEM:
✓ Core ML Engine (forecasting, features)
✓ Elite Consulting Frameworks (BCG, Porter, Working Capital, etc.)
✓ Strategic Report Generation
✓ Dashboard Integration Ready

This file REPLACES both kustawi_ml_engine.py and elite_consulting_engine.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import hashlib
import json
import joblib
import logging
import warnings
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

# ============================================================================
# DATA PROTECTION COMPLIANCE
# ============================================================================

class DataProtectionCompliance:
    """Kenya Data Protection Act 2019 Compliance Layer"""
    
    @staticmethod
    def anonymize_customer_data(df):
        """Remove/hash PII as per DPA 2019 Section 25"""
        pii_columns = ['Customer ID', 'Customer Name', 'Customer_ID', 'Customer_Name']
        
        for col in pii_columns:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:16]
                )
        return df
    
    @staticmethod
    def log_data_processing(operation, data_count):
        """Audit trail as per DPA 2019 Section 27"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'records_processed': data_count,
            'user': 'kustawi_system',
            'compliance': 'DPA_2019'
        }
        
        try:
            with open('kustawi_audit_log.json', 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except:
            pass

# ============================================================================
# CORE ML ENGINE
# ============================================================================

class PredictaKenyaEngine:
    """
    Unified AI & Consulting Engine
    Combines ML forecasting with strategic consulting frameworks
    """
    
    VERSION = "3.0.0"
    COPYRIGHT = "Kustawi Digital Solutions Ltd"
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.scaler_stats = None
        self.compliance = DataProtectionCompliance()
        self.training_history = []
    
    def load_and_validate_data(self, df):
        """Load and validate data with compliance checks"""
        df = self.compliance.anonymize_customer_data(df)
        
        required_cols = ['Date', 'Sales']
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date', 'Sales'])
        df = df.drop_duplicates()
        
        self.compliance.log_data_processing('data_load', len(df))
        
        return df.sort_values('Date').reset_index(drop=True)
    
    def kenyan_seasonality_detection(self, df):
        """PROPRIETARY: Kenyan Market Seasonality Detection"""
        df = df.copy()
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        
        df['Is_Christmas_Season'] = df['Month'].isin([11, 12]).astype(int)
        df['Is_Easter_Period'] = df['Month'].isin([3, 4]).astype(int)
        df['Is_School_Opening'] = df['Month'].isin([1, 5, 9]).astype(int)
        df['Is_Payday_Period'] = ((df['Day'] >= 25) | (df['Day'] <= 5)).astype(int)
        df['Is_Quarter_End'] = df['Month'].isin([3, 6, 9, 12]).astype(int)
        
        return df
    
    def advanced_feature_engineering(self, df):
        """Advanced feature engineering with 40+ features"""
        df['YearMonth'] = df['Date'].dt.to_period('M')
        
        monthly_sales = df.groupby('YearMonth').agg({
            'Sales': 'sum',
            'Date': 'first'
        }).reset_index()
        
        monthly_sales.columns = ['YearMonth', 'Sales', 'Date']
        monthly_sales['Date'] = monthly_sales['YearMonth'].dt.to_timestamp()
        monthly_sales = monthly_sales.sort_values('Date').reset_index(drop=True)
        
        monthly_sales = self.kenyan_seasonality_detection(monthly_sales)
        
        monthly_sales['Year'] = monthly_sales['Date'].dt.year
        monthly_sales['Quarter'] = monthly_sales['Date'].dt.quarter
        monthly_sales['DayOfYear'] = monthly_sales['Date'].dt.dayofyear
        monthly_sales['WeekOfYear'] = monthly_sales['Date'].dt.isocalendar().week
        
        monthly_sales['Month_Sin'] = np.sin(2 * np.pi * monthly_sales['Month'] / 12)
        monthly_sales['Month_Cos'] = np.cos(2 * np.pi * monthly_sales['Month'] / 12)
        
        for lag in [1, 2, 3, 6, 12]:
            monthly_sales[f'Sales_Lag_{lag}'] = monthly_sales['Sales'].shift(lag)
        
        for window in [3, 6, 12]:
            monthly_sales[f'Sales_RollMean_{window}'] = monthly_sales['Sales'].rolling(window=window).mean()
            monthly_sales[f'Sales_RollStd_{window}'] = monthly_sales['Sales'].rolling(window=window).std()
            monthly_sales[f'Sales_RollMin_{window}'] = monthly_sales['Sales'].rolling(window=window).min()
            monthly_sales[f'Sales_RollMax_{window}'] = monthly_sales['Sales'].rolling(window=window).max()
        
        monthly_sales['Sales_EMA_3'] = monthly_sales['Sales'].ewm(span=3, adjust=False).mean()
        monthly_sales['Sales_EMA_6'] = monthly_sales['Sales'].ewm(span=6, adjust=False).mean()
        monthly_sales['Trend'] = range(len(monthly_sales))
        monthly_sales['Sales_GrowthRate'] = monthly_sales['Sales'].pct_change()
        
        if len(monthly_sales) > 12:
            monthly_sales['Sales_YoY_Change'] = monthly_sales['Sales'].pct_change(periods=12)
        
        monthly_sales['Sales_Volatility'] = monthly_sales['Sales'].rolling(window=6).std() / monthly_sales['Sales'].rolling(window=6).mean()
        
        monthly_sales_clean = monthly_sales.dropna().reset_index(drop=True)
        
        return monthly_sales_clean
    
    def train_model(self, df, test_size=0.2):
        """Train ensemble model with performance tracking"""
        
        feature_cols = [col for col in df.columns if col not in ['Date', 'YearMonth', 'Sales']]
        
        split_idx = int(len(df) * (1 - test_size))
        
        train = df[:split_idx].copy()
        test = df[split_idx:].copy()
        
        X_train = train[feature_cols]
        y_train = train['Sales']
        X_test = test[feature_cols]
        y_test = test['Sales']
        
        gbr = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            subsample=0.8
        )
        
        gbr.fit(X_train, y_train)
        
        train_pred = gbr.predict(X_train)
        test_pred = gbr.predict(X_test)
        
        metrics = {
            'train_rmse': float(np.sqrt(mean_squared_error(y_train, train_pred))),
            'test_rmse': float(np.sqrt(mean_squared_error(y_test, test_pred))),
            'rmse': float(np.sqrt(mean_squared_error(y_test, test_pred))),
            'train_r2': float(r2_score(y_train, train_pred)),
            'test_r2': float(r2_score(y_test, test_pred)),
            'r2': float(r2_score(y_test, test_pred)),
            'mae': float(mean_absolute_error(y_test, test_pred)),
            'mape': float(np.mean(np.abs((y_test - test_pred) / y_test)) * 100),
            'residual_std': float(np.std(y_test.values - test_pred))
        }
        
        self.model = gbr
        self.feature_names = feature_cols
        self.scaler_stats = {
            'last_12_months_sales': df['Sales'].tail(12).tolist(),
            'last_date': df['Date'].max(),
            'training_samples': len(train),
            'test_samples': len(test),
            'residual_std': metrics['residual_std']
        }
        
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'samples': len(train)
        })
        
        self.compliance.log_data_processing('model_training', len(train))
        
        return metrics
    
    def generate_forecast(self, df, periods=12):
        """Generate multi-horizon forecast"""
        
        if self.model is None:
            raise ValueError("Model not trained")
        
        last_date = df['Date'].max()
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=periods,
            freq='MS'
        )
        
        future_features_list = []
        
        for idx, date in enumerate(future_dates):
            row = {
                'Year': date.year,
                'Month': date.month,
                'Quarter': date.quarter,
                'Day': date.day,
                'DayOfWeek': date.dayofweek,
                'DayOfYear': date.dayofyear,
                'WeekOfYear': date.isocalendar().week,
                'Is_Christmas_Season': int(date.month in [11, 12]),
                'Is_Easter_Period': int(date.month in [3, 4]),
                'Is_School_Opening': int(date.month in [1, 5, 9]),
                'Is_Payday_Period': int((date.day >= 25) or (date.day <= 5)),
                'Is_Quarter_End': int(date.month in [3, 6, 9, 12]),
                'Month_Sin': np.sin(2 * np.pi * date.month / 12),
                'Month_Cos': np.cos(2 * np.pi * date.month / 12),
                'Sales_Lag_1': df['Sales'].iloc[-1],
                'Sales_Lag_2': df['Sales'].iloc[-2],
                'Sales_Lag_3': df['Sales'].iloc[-3],
                'Sales_Lag_6': df['Sales'].iloc[-6],
                'Sales_Lag_12': df['Sales'].iloc[-12],
                'Sales_RollMean_3': df['Sales'].iloc[-3:].mean(),
                'Sales_RollStd_3': df['Sales'].iloc[-3:].std(),
                'Sales_RollMin_3': df['Sales'].iloc[-3:].min(),
                'Sales_RollMax_3': df['Sales'].iloc[-3:].max(),
                'Sales_RollMean_6': df['Sales'].iloc[-6:].mean(),
                'Sales_RollStd_6': df['Sales'].iloc[-6:].std(),
                'Sales_RollMin_6': df['Sales'].iloc[-6:].min(),
                'Sales_RollMax_6': df['Sales'].iloc[-6:].max(),
                'Sales_RollMean_12': df['Sales'].iloc[-12:].mean(),
                'Sales_RollStd_12': df['Sales'].iloc[-12:].std(),
                'Sales_RollMin_12': df['Sales'].iloc[-12:].min(),
                'Sales_RollMax_12': df['Sales'].iloc[-12:].max(),
                'Sales_EMA_3': df['Sales'].ewm(span=3, adjust=False).mean().iloc[-1],
                'Sales_EMA_6': df['Sales'].ewm(span=6, adjust=False).mean().iloc[-1],
                'Trend': len(df) + idx + 1,
                'Sales_GrowthRate': df['Sales'].pct_change().mean(),
                'Sales_YoY_Change': df['Sales'].pct_change(12).mean() if len(df) > 12 else 0,
                'Sales_Volatility': df['Sales'].iloc[-6:].std() / df['Sales'].iloc[-6:].mean()
            }
            future_features_list.append(row)
        
        future_features_df = pd.DataFrame(future_features_list)
        future_features_df = future_features_df[self.feature_names]
        
        forecast_values = self.model.predict(future_features_df)
        
        base_std = self.scaler_stats.get('residual_std', np.std(forecast_values) * 0.1)
        confidence_multiplier = 1.96
        uncertainty_factor = np.linspace(1.0, 1.5, periods)
        
        lower_bounds = forecast_values - (confidence_multiplier * base_std * uncertainty_factor)
        upper_bounds = forecast_values + (confidence_multiplier * base_std * uncertainty_factor)
        
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'YearMonth': future_dates.strftime('%Y-%m'),
            'Forecast': forecast_values,
            'Lower_Bound': lower_bounds,
            'Upper_Bound': upper_bounds,
            'Confidence': 95
        })
        
        self.compliance.log_data_processing('forecast_generation', periods)
        
        return forecast_df
    
    def save_model(self, filepath='predictakenya_model.pkl'):
        """Save model for dashboard integration"""
        
        if self.model is None:
            raise ValueError("No model to save")
        
        model_package = {
            'model': self.model,
            'feature_names': self.feature_names,
            'scaler_stats': self.scaler_stats,
            'training_history': self.training_history,
            'version': self.VERSION,
            'copyright': self.COPYRIGHT,
            'trained_date': datetime.now().isoformat()
        }
        
        joblib.dump(model_package, filepath)
        
        return filepath

# ============================================================================
# ELITE CONSULTING FRAMEWORKS
# ============================================================================

class StrategicConsultingEngine:
    """Elite-level consulting analysis - McKinsey/BCG frameworks"""
    
    @staticmethod
    def bcg_matrix_analysis(df: pd.DataFrame) -> Dict:
        """BCG Growth-Share Matrix"""
        
        if 'Product' not in df.columns:
            return {'available': False}
        
        df['YearMonth'] = df['Date'].dt.to_period('M')
        monthly_product = df.groupby(['YearMonth', 'Product'])['Sales'].sum().reset_index()
        
        product_analysis = []
        
        for product in df['Product'].unique():
            product_data = monthly_product[monthly_product['Product'] == product]
            
            if len(product_data) >= 12:
                recent_12m = product_data.tail(12)['Sales'].sum()
                previous_12m = product_data.iloc[-24:-12]['Sales'].sum() if len(product_data) >= 24 else recent_12m
                
                growth_rate = ((recent_12m - previous_12m) / previous_12m * 100) if previous_12m > 0 else 0
                
                total_sales = df.groupby('Product')['Sales'].sum()
                market_share = (total_sales[product] / total_sales.sum()) * 100
                
                if 'Profit' in df.columns:
                    profit_margin = (df[df['Product'] == product]['Profit'].sum() / 
                                   df[df['Product'] == product]['Sales'].sum()) * 100
                else:
                    profit_margin = 15
                
                high_growth_threshold = df.groupby('Product').apply(
                    lambda x: ((x.tail(12)['Sales'].sum() - x.iloc[-24:-12]['Sales'].sum()) / 
                              x.iloc[-24:-12]['Sales'].sum() * 100) if len(x) >= 24 else 0
                ).median()
                
                high_share_threshold = total_sales.quantile(0.5) / total_sales.sum() * 100
                
                if growth_rate > high_growth_threshold and market_share > high_share_threshold:
                    category = "⭐ STAR"
                    strategy = "INVEST HEAVILY - Scale production, expand distribution"
                    priority = 1
                elif growth_rate <= high_growth_threshold and market_share > high_share_threshold:
                    category = "💰 CASH COW"
                    strategy = "HARVEST - Maximize margins, use cash for Stars"
                    priority = 2
                elif growth_rate > high_growth_threshold and market_share <= high_share_threshold:
                    category = "❓ QUESTION MARK"
                    strategy = "SELECTIVE - Test market, invest or divest"
                    priority = 3
                else:
                    category = "🐕 DOG"
                    strategy = "DIVEST - Phase out, reallocate resources"
                    priority = 4
                
                product_analysis.append({
                    'Product': product,
                    'Category': category,
                    'Growth_Rate': growth_rate,
                    'Market_Share': market_share,
                    'Profit_Margin': profit_margin,
                    'Annual_Revenue': recent_12m,
                    'Strategy': strategy,
                    'Priority': priority
                })
        
        return {
            'available': True,
            'analysis': pd.DataFrame(product_analysis).sort_values('Priority') if product_analysis else pd.DataFrame(),
            'summary': {
                'stars': len([p for p in product_analysis if '⭐' in p['Category']]),
                'cash_cows': len([p for p in product_analysis if '💰' in p['Category']]),
                'question_marks': len([p for p in product_analysis if '❓' in p['Category']]),
                'dogs': len([p for p in product_analysis if '🐕' in p['Category']])
            }
        }
    
    @staticmethod
    def working_capital_optimization(df: pd.DataFrame, forecast_df: pd.DataFrame) -> Dict:
        """CFO-Level Financial Analysis"""
        
        df['YearMonth'] = df['Date'].dt.to_period('M')
        monthly_sales = df.groupby('YearMonth')['Sales'].sum()
        
        avg_monthly_revenue = monthly_sales.mean()
        
        if 'Quantity' in df.columns:
            avg_inventory_value = df['Sales'].sum() / df['Quantity'].sum() * df.groupby('Product')['Quantity'].mean().mean() if 'Product' in df.columns else avg_monthly_revenue * 0.5
        else:
            avg_inventory_value = avg_monthly_revenue * 0.5
        
        inventory_turnover = (monthly_sales.sum() / avg_inventory_value) if avg_inventory_value > 0 else 12
        days_inventory_outstanding = 365 / inventory_turnover
        
        days_sales_outstanding = 30
        days_payable_outstanding = 60
        
        cash_conversion_cycle = days_inventory_outstanding + days_sales_outstanding - days_payable_outstanding
        
        annual_revenue_forecast = forecast_df['Forecast'].sum()
        working_capital_required = (cash_conversion_cycle / 365) * annual_revenue_forecast
        
        optimized_dio = days_inventory_outstanding * 0.8
        optimized_dso = 25
        optimized_dpo = 75
        
        optimized_ccc = optimized_dio + optimized_dso - optimized_dpo
        optimized_wc = (optimized_ccc / 365) * annual_revenue_forecast
        
        cash_freed = working_capital_required - optimized_wc
        
        return {
            'available': True,
            'current': {
                'DIO': days_inventory_outstanding,
                'DSO': days_sales_outstanding,
                'DPO': days_payable_outstanding,
                'CCC': cash_conversion_cycle,
                'Working_Capital_Required': working_capital_required
            },
            'optimized': {
                'DIO': optimized_dio,
                'DSO': optimized_dso,
                'DPO': optimized_dpo,
                'CCC': optimized_ccc,
                'Working_Capital_Required': optimized_wc
            },
            'opportunity': {
                'Cash_Freed': cash_freed,
                'ROI_Potential': cash_freed * 0.15,
                'Implementation_Actions': [
                    f"Reduce inventory by {(days_inventory_outstanding - optimized_dio):.0f} days → Free KES {cash_freed * 0.6:,.0f}",
                    f"Accelerate collections to 25 days → Improve cash flow KES {cash_freed * 0.2:,.0f}",
                    f"Negotiate supplier terms to 75 days → Free up KES {cash_freed * 0.2:,.0f}"
                ]
            }
        }
    
    @staticmethod
    def competitive_positioning(df: pd.DataFrame, forecast_df: pd.DataFrame) -> Dict:
        """Porter's Five Forces + Competitive Strategy"""
        
        total_revenue = df['Sales'].sum()
        revenue_growth = ((forecast_df['Forecast'].sum() - total_revenue) / total_revenue) * 100
        
        market_growth_rate = 12  # Kenyan retail market ~12% annually
        performance_vs_market = revenue_growth - market_growth_rate
        
        if 'Product' in df.columns:
            product_shares = df.groupby('Product')['Sales'].sum() / total_revenue
            herfindahl_index = (product_shares ** 2).sum()
            diversification_score = 1 - herfindahl_index
        else:
            diversification_score = 0.5
        
        if performance_vs_market > 5:
            position = "MARKET LEADER"
            strategy = "Aggressive expansion - Open locations, acquire competitors"
        elif performance_vs_market > 0:
            position = "STRONG PERFORMER"
            strategy = "Selective growth - Focus on high-margin products"
        elif performance_vs_market > -5:
            position = "MARKET FOLLOWER"
            strategy = "Defensive - Improve efficiency, differentiate"
        else:
            position = "UNDERPERFORMER"
            strategy = "Turnaround - Restructure, divest unprofitable lines"
        
        return {
            'available': True,
            'position': position,
            'growth_vs_market': performance_vs_market,
            'revenue_growth': revenue_growth,
            'market_growth': market_growth_rate,
            'diversification_score': diversification_score,
            'strategy': strategy,
            'five_forces': {
                'Competitive_Rivalry': 'HIGH',
                'Supplier_Power': 'MEDIUM',
                'Buyer_Power': 'HIGH',
                'Threat_of_New_Entrants': 'MEDIUM',
                'Threat_of_Substitutes': 'LOW'
            }
        }

# ============================================================================
# UNIFIED ANALYTICS
# ============================================================================

class UnifiedAnalytics:
    """
    Combines all analytics in one place
    Used by dashboard and Colab notebook
    """
    
    def __init__(self):
        self.engine = PredictaKenyaEngine()
        self.consultant = StrategicConsultingEngine()
    
    def run_complete_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Run complete analysis: ML + Consulting
        Returns all insights for dashboard
        """
        
        # Core ML
        df_validated = self.engine.load_and_validate_data(df)
        df_features = self.engine.advanced_feature_engineering(df_validated)
        metrics = self.engine.train_model(df_features, test_size=0.2)
        forecast = self.engine.generate_forecast(df_features, periods=12)
        
        # Strategic Consulting
        bcg = self.consultant.bcg_matrix_analysis(df)
        wc = self.consultant.working_capital_optimization(df, forecast)
        competitive = self.consultant.competitive_positioning(df, forecast)
        
        # Package everything
        return {
            'data': df_validated,
            'features': df_features,
            'metrics': metrics,
            'forecast': forecast,
            'consulting': {
                'bcg': bcg,
                'working_capital': wc,
                'competitive': competitive
            }
        }

# Export main class
__all__ = ['PredictaKenyaEngine', 'StrategicConsultingEngine', 'UnifiedAnalytics', 'DataProtectionCompliance']
