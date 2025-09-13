#!/usr/bin/env python3
"""
Feature Importance Visualization Tools
=====================================

A comprehensive set of visualization tools for feature importance analysis,
including heatmaps, bar charts, temporal trend plots, and interactive
exploration utilities.

This module provides:
- Static and interactive visualizations
- Temporal importance trend plots
- Feature comparison utilities
- Drift visualization
- Interactive dashboards

Author: AlgoTrading System
Version: 1.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available - interactive visualizations will be disabled")

try:
    from bokeh.plotting import figure, show
    from bokeh.models import ColumnDataSource, HoverTool, Div
    from bokeh.layouts import column, row
    from bokeh.io import output_notebook
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False
    print("Bokeh not available - some interactive features will be disabled")

from .importance_engine import ImportanceResult, ImportanceScore, ImportanceMethod
from .importance_tracker import ImportanceTracker, DriftAlert, ImportanceTrend

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ImportanceVisualizer:
    """Main class for feature importance visualizations."""

    def __init__(self,
                 tracker: Optional[ImportanceTracker] = None,
                 output_dir: str = "importance_visualizations"):
        """
        Initialize importance visualizer.

        Args:
            tracker: ImportanceTracker instance for data access
            output_dir: Directory to save visualizations
        """
        self.tracker = tracker
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Set up plotting parameters
        self.color_palette = sns.color_palette("husl", 12)
        self.figure_size = (12, 8)
        self.dpi = 300

    def create_importance_bar_chart(self,
                                  result: ImportanceResult,
                                  top_n: int = 20,
                                  title: Optional[str] = None,
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Create horizontal bar chart of feature importance.

        Args:
            result: Importance calculation result
            top_n: Number of top features to show
            title: Custom title for the chart
            save_path: Path to save the figure

        Returns:
            Matplotlib figure
        """
        # Prepare data
        feature_scores = [(score.feature_name, score.score) for score in result.scores]
        feature_scores.sort(key=lambda x: x[1], reverse=True)

        if top_n and len(feature_scores) > top_n:
            feature_scores = feature_scores[:top_n]

        features, scores = zip(*feature_scores)

        # Create figure
        fig, ax = plt.subplots(figsize=self.figure_size)

        bars = ax.barh(range(len(features)), scores, color=self.color_palette[0])

        # Customize plot
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('Importance Score')
        ax.set_title(title or f'Feature Importance - {result.method.value.title()}')

        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(bar.get_width() + 0.01 * max(scores), bar.get_y() + bar.get_height()/2,
                   f'{score:.3f}', va='center', ha='left')

        # Add grid
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Bar chart saved to {save_path}")

        return fig

    def create_importance_heatmap(self,
                                 results: Dict[ImportanceMethod, ImportanceResult],
                                 normalize: bool = True,
                                 title: Optional[str] = None,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Create heatmap comparing importance across multiple methods.

        Args:
            results: Dictionary of importance results by method
            normalize: Whether to normalize scores
            title: Custom title for the heatmap
            save_path: Path to save the figure

        Returns:
            Matplotlib figure
        """
        if not results:
            raise ValueError("No importance results provided")

        # Prepare data matrix
        all_features = set()
        for result in results.values():
            all_features.update(score.feature_name for score in result.scores)

        all_features = sorted(all_features)
        method_names = [method.value for method in results.keys()]

        # Create importance matrix
        importance_matrix = np.zeros((len(all_features), len(results)))

        for j, (method, result) in enumerate(results.items()):
            method_scores = {score.feature_name: score.score for score in result.scores}

            for i, feature in enumerate(all_features):
                importance_matrix[i, j] = method_scores.get(feature, 0.0)

        # Normalize if requested
        if normalize:
            importance_matrix = importance_matrix / importance_matrix.max(axis=0, keepdims=True)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(max(10, len(method_names) * 2), max(8, len(all_features) * 0.3)))

        im = ax.imshow(importance_matrix, cmap='YlOrRd', aspect='auto')

        # Customize plot
        ax.set_xticks(range(len(method_names)))
        ax.set_xticklabels(method_names, rotation=45, ha='right')
        ax.set_yticks(range(len(all_features)))
        ax.set_yticklabels(all_features)
        ax.set_title(title or 'Feature Importance Comparison Across Methods')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Importance Score' + (' (Normalized)' if normalize else ''))

        # Add text annotations
        for i in range(len(all_features)):
            for j in range(len(method_names)):
                value = importance_matrix[i, j]
                if value > 0.01:  # Only show significant values
                    ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                           color='white' if value > 0.5 else 'black', fontsize=8)

        plt.tight_layout()

        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Heatmap saved to {save_path}")

        return fig

    def create_temporal_trend_plot(self,
                                 feature_names: List[str],
                                 start_date: Optional[datetime] = None,
                                 end_date: Optional[datetime] = None,
                                 methods: Optional[List[ImportanceMethod]] = None,
                                 title: Optional[str] = None,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Create temporal trend plot for feature importance.

        Args:
            feature_names: List of features to plot
            start_date: Start date for the plot
            end_date: End date for the plot
            methods: Methods to include
            title: Custom title for the plot
            save_path: Path to save the figure

        Returns:
            Matplotlib figure
        """
        if self.tracker is None:
            raise ValueError("ImportanceTracker is required for temporal plots")

        # Get historical data
        history_df = self.tracker.get_importance_history(
            feature_names=feature_names,
            start_time=start_date,
            end_time=end_date,
            methods=methods
        )

        if history_df.empty:
            logger.warning("No historical data available for temporal plot")
            fig, ax = plt.subplots(figsize=self.figure_size)
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            return fig

        # Create figure
        fig, ax = plt.subplots(figsize=self.figure_size)

        # Plot each feature
        for i, feature in enumerate(feature_names):
            feature_data = history_df[history_df['feature_name'] == feature]

            if not feature_data.empty:
                color = self.color_palette[i % len(self.color_palette)]
                ax.plot(feature_data['timestamp'], feature_data['importance_score'],
                       marker='o', linewidth=2, markersize=4, label=feature, color=color)

                # Add trend line
                if len(feature_data) > 5:
                    x_numeric = mdates.date2num(feature_data['timestamp'])
                    z = np.polyfit(x_numeric, feature_data['importance_score'], 1)
                    p = np.poly1d(z)
                    ax.plot(feature_data['timestamp'], p(x_numeric), '--', alpha=0.7, color=color)

        # Customize plot
        ax.set_xlabel('Date')
        ax.set_ylabel('Importance Score')
        ax.set_title(title or 'Feature Importance Trends Over Time')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        # Add grid
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Temporal trend plot saved to {save_path}")

        return fig

    def create_drift_alert_plot(self,
                               alerts: List[DriftAlert],
                               title: Optional[str] = None,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Create visualization of drift alerts.

        Args:
            alerts: List of drift alerts
            title: Custom title for the plot
            save_path: Path to save the figure

        Returns:
            Matplotlib figure
        """
        if not alerts:
            fig, ax = plt.subplots(figsize=self.figure_size)
            ax.text(0.5, 0.5, 'No drift alerts to display', ha='center', va='center', transform=ax.transAxes)
            return fig

        # Prepare data
        alert_data = []
        for alert in alerts:
            alert_data.append({
                'feature': alert.feature_name,
                'timestamp': alert.timestamp,
                'severity': alert.severity.value,
                'drift_score': alert.drift_score,
                'method': alert.method.value
            })

        df = pd.DataFrame(alert_data)

        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Alerts over time
        severity_colors = {'none': 'gray', 'low': 'yellow', 'medium': 'orange', 'high': 'red', 'critical': 'darkred'}
        for severity in df['severity'].unique():
            severity_data = df[df['severity'] == severity]
            ax1.scatter(severity_data['timestamp'], severity_data['feature'],
                       c=severity_colors.get(severity, 'gray'), s=100, alpha=0.7, label=severity.title())

        ax1.set_xlabel('Date')
        ax1.set_ylabel('Feature')
        ax1.set_title('Drift Alerts Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Severity distribution
        severity_counts = df['severity'].value_counts()
        colors = [severity_colors.get(sev, 'gray') for sev in severity_counts.index]
        ax2.pie(severity_counts.values, labels=severity_counts.index, autopct='%1.1f%%', colors=colors)
        ax2.set_title('Alert Severity Distribution')

        # Plot 3: Drift scores by feature
        for feature in df['feature'].unique():
            feature_data = df[df['feature'] == feature]
            ax3.scatter(feature_data['timestamp'], feature_data['drift_score'],
                       label=feature, s=100, alpha=0.7)

        ax3.set_xlabel('Date')
        ax3.set_ylabel('Drift Score')
        ax3.set_title('Drift Scores Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Method usage
        method_counts = df['method'].value_counts()
        ax4.bar(method_counts.index, method_counts.values, color=self.color_palette[0])
        ax4.set_xlabel('Detection Method')
        ax4.set_ylabel('Number of Alerts')
        ax4.set_title('Alerts by Detection Method')
        ax4.tick_params(axis='x', rotation=45)

        plt.suptitle(title or 'Feature Importance Drift Analysis', fontsize=16)
        plt.tight_layout()

        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Drift alert plot saved to {save_path}")

        return fig

    def create_correlation_matrix(self,
                                feature_names: List[str],
                                start_date: Optional[datetime] = None,
                                end_date: Optional[datetime] = None,
                                title: Optional[str] = None,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Create correlation matrix of feature importance over time.

        Args:
            feature_names: List of features to analyze
            start_date: Start date for analysis
            end_date: End date for analysis
            title: Custom title for the plot
            save_path: Path to save the figure

        Returns:
            Matplotlib figure
        """
        if self.tracker is None:
            raise ValueError("ImportanceTracker is required for correlation analysis")

        # Get historical data and pivot to wide format
        history_df = self.tracker.get_importance_history(
            feature_names=feature_names,
            start_time=start_date,
            end_time=end_date
        )

        if history_df.empty or len(feature_names) < 2:
            fig, ax = plt.subplots(figsize=self.figure_size)
            ax.text(0.5, 0.5, 'Insufficient data for correlation analysis', ha='center', va='center', transform=ax.transAxes)
            return fig

        # Pivot data to create correlation matrix
        pivot_df = history_df.pivot_table(
            index='timestamp',
            columns='feature_name',
            values='importance_score',
            aggfunc='mean'
        )

        # Calculate correlation matrix
        correlation_matrix = pivot_df.corr()

        # Create heatmap
        fig, ax = plt.subplots(figsize=(max(8, len(feature_names) * 0.8), max(6, len(feature_names) * 0.8)))

        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax, fmt='.2f', cbar_kws={"shrink": 0.8})

        ax.set_title(title or 'Feature Importance Correlation Matrix')

        plt.tight_layout()

        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Correlation matrix saved to {save_path}")

        return fig

    def create_feature_radar_chart(self,
                                  results: Dict[ImportanceMethod, ImportanceResult],
                                  feature_names: List[str],
                                  normalize: bool = True,
                                  title: Optional[str] = None,
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Create radar chart comparing feature importance across methods.

        Args:
            results: Dictionary of importance results by method
            feature_names: Features to include in radar chart
            normalize: Whether to normalize scores
            title: Custom title for the chart
            save_path: Path to save the figure

        Returns:
            Matplotlib figure
        """
        if not results:
            raise ValueError("No importance results provided")

        # Prepare data
        method_names = [method.value for method in results.keys()]
        colors = self.color_palette[:len(method_names)]

        # Prepare angles for radar chart
        angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        # Create figure
        fig, ax = plt.subplots(figsize=self.figure_size, subplot_kw=dict(projection='polar'))

        # Plot each method
        for i, (method, result) in enumerate(results.items()):
            scores = []
            for feature in feature_names:
                feature_scores = [s.score for s in result.scores if s.feature_name == feature]
                avg_score = np.mean(feature_scores) if feature_scores else 0.0
                scores.append(avg_score)

            scores += scores[:1]  # Complete the circle

            if normalize and max(scores[:-1]) > 0:
                scores = [s / max(scores[:-1]) for s in scores]

            ax.plot(angles, scores, 'o-', linewidth=2, label=method.value, color=colors[i])
            ax.fill(angles, scores, alpha=0.25, color=colors[i])

        # Customize plot
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_names)
        ax.set_ylim(0, 1.1 if normalize else None)
        ax.set_title(title or 'Feature Importance Radar Chart')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        plt.tight_layout()

        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Radar chart saved to {save_path}")

        return fig

    def create_importance_distribution_plot(self,
                                          results: Dict[ImportanceMethod, ImportanceResult],
                                          title: Optional[str] = None,
                                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Create distribution plots for importance scores.

        Args:
            results: Dictionary of importance results by method
            title: Custom title for the plot
            save_path: Path to save the figure

        Returns:
            Matplotlib figure
        """
        if not results:
            raise ValueError("No importance results provided")

        # Create figure
        n_methods = len(results)
        n_cols = min(3, n_methods)
        n_rows = (n_methods + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_methods == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()

        # Plot each method
        for i, (method, result) in enumerate(results.items()):
            ax = axes[i]
            scores = [score.score for score in result.scores]

            # Create histogram with KDE
            ax.hist(scores, bins=30, density=True, alpha=0.7, color=self.color_palette[i], edgecolor='black')

            # Add KDE if possible
            try:
                from scipy import stats
                kde = stats.gaussian_kde(scores)
                x_range = np.linspace(min(scores), max(scores), 100)
                ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
            except ImportError:
                pass

            ax.set_xlabel('Importance Score')
            ax.set_ylabel('Density')
            ax.set_title(f'{method.value.title()} Distribution')
            ax.grid(True, alpha=0.3)

        # Remove unused subplots
        for i in range(n_methods, len(axes)):
            fig.delaxes(axes[i])

        plt.suptitle(title or 'Feature Importance Distributions', fontsize=16)
        plt.tight_layout()

        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Distribution plot saved to {save_path}")

        return fig

    def create_summary_dashboard(self,
                               feature_names: List[str],
                               lookback_days: int = 30,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive summary dashboard.

        Args:
            feature_names: Features to include in dashboard
            lookback_days: Number of days to look back
            save_path: Path to save the figure

        Returns:
            Matplotlib figure
        """
        if self.tracker is None:
            raise ValueError("ImportanceTracker is required for dashboard creation")

        # Get data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        history_df = self.tracker.get_importance_history(
            feature_names=feature_names,
            start_time=start_date,
            end_time=end_date
        )

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Subplot 1: Top features bar chart
        ax1 = fig.add_subplot(gs[0, 0])
        if not history_df.empty:
            top_features = history_df.groupby('feature_name')['importance_score'].mean().nlargest(10)
            top_features.plot(kind='barh', ax=ax1, color=self.color_palette[0])
            ax1.set_title('Top 10 Features by Average Importance')
            ax1.set_xlabel('Average Importance Score')

        # Subplot 2: Trend over time
        ax2 = fig.add_subplot(gs[0, 1])
        if not history_df.empty:
            daily_avg = history_df.groupby(history_df['timestamp'].dt.date)['importance_score'].mean()
            ax2.plot(daily_avg.index, daily_avg.values, marker='o', linewidth=2)
            ax2.set_title('Average Importance Trend')
            ax2.set_ylabel('Average Importance')
            ax2.tick_params(axis='x', rotation=45)

        # Subplot 3: Feature count over time
        ax3 = fig.add_subplot(gs[0, 2])
        if not history_df.empty:
            daily_features = history_df.groupby(history_df['timestamp'].dt.date)['feature_name'].nunique()
            ax3.plot(daily_features.index, daily_features.values, marker='o', linewidth=2, color=self.color_palette[1])
            ax3.set_title('Number of Features Analyzed')
            ax3.set_ylabel('Feature Count')
            ax3.tick_params(axis='x', rotation=45)

        # Subplot 4: Method usage
        ax4 = fig.add_subplot(gs[1, 0])
        if not history_df.empty:
            method_counts = history_df['method'].value_counts()
            ax4.pie(method_counts.values, labels=method_counts.index, autopct='%1.1f%%')
            ax4.set_title('Importance Method Usage')

        # Subplot 5: Recent drift alerts
        ax5 = fig.add_subplot(gs[1, 1])
        recent_alerts = self.tracker.alerts
        if recent_alerts:
            alert_df = pd.DataFrame([{
                'timestamp': alert.timestamp,
                'severity': alert.severity.value,
                'feature': alert.feature_name
            } for alert in list(recent_alerts)[-50:]])  # Last 50 alerts

            severity_counts = alert_df['severity'].value_counts()
            severity_counts.plot(kind='bar', ax=ax5, color=['red', 'orange', 'yellow'])
            ax5.set_title('Recent Drift Alerts by Severity')
            ax5.set_ylabel('Count')
            ax5.tick_params(axis='x', rotation=45)
        else:
            ax5.text(0.5, 0.5, 'No recent alerts', ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Recent Drift Alerts')

        # Subplot 6: Importance volatility
        ax6 = fig.add_subplot(gs[1, 2])
        if not history_df.empty:
            volatility = history_df.groupby('feature_name')['importance_score'].std() / history_df.groupby('feature_name')['importance_score'].mean()
            volatility = volatility.dropna().sort_values(ascending=False).head(10)
            volatility.plot(kind='bar', ax=ax6, color=self.color_palette[2])
            ax6.set_title('Top 10 Most Volatile Features')
            ax6.set_ylabel('Volatility (CV)')
            ax6.tick_params(axis='x', rotation=45)

        # Subplot 7: Feature correlation heatmap
        ax7 = fig.add_subplot(gs[2, :])
        if not history_df.empty and len(feature_names) >= 2:
            pivot_df = history_df.pivot_table(
                index='timestamp',
                columns='feature_name',
                values='importance_score',
                aggfunc='mean'
            )

            if len(pivot_df.columns) >= 2:
                corr_matrix = pivot_df.corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                           square=True, ax=ax7, fmt='.2f', cbar_kws={"shrink": 0.6})
                ax7.set_title('Feature Importance Correlation Matrix')
            else:
                ax7.text(0.5, 0.5, 'Insufficient features for correlation analysis',
                         ha='center', va='center', transform=ax7.transAxes)
        else:
            ax7.text(0.5, 0.5, 'Insufficient data for correlation analysis',
                     ha='center', va='center', transform=ax7.transAxes)

        plt.suptitle(f'Feature Importance Dashboard - Last {lookback_days} Days', fontsize=18)

        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Dashboard saved to {save_path}")

        return fig


class InteractiveImportanceVisualizer:
    """Interactive visualization tools using Plotly/Bokeh."""

    def __init__(self, tracker: Optional[ImportanceTracker] = None):
        """
        Initialize interactive visualizer.

        Args:
            tracker: ImportanceTracker instance for data access
        """
        self.tracker = tracker

        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available - interactive visualizations disabled")

    def create_interactive_trend_plot(self,
                                    feature_names: List[str],
                                    start_date: Optional[datetime] = None,
                                    end_date: Optional[datetime] = None) -> Optional[go.Figure]:
        """
        Create interactive trend plot using Plotly.

        Args:
            feature_names: Features to plot
            start_date: Start date for the plot
            end_date: End date for the plot

        Returns:
            Plotly figure or None if Plotly not available
        """
        if not PLOTLY_AVAILABLE or self.tracker is None:
            return None

        # Get historical data
        history_df = self.tracker.get_importance_history(
            feature_names=feature_names,
            start_time=start_date,
            end_time=end_date
        )

        if history_df.empty:
            return None

        # Create figure
        fig = go.Figure()

        # Add traces for each feature
        for feature in feature_names:
            feature_data = history_df[history_df['feature_name'] == feature]

            if not feature_data.empty:
                fig.add_trace(go.Scatter(
                    x=feature_data['timestamp'],
                    y=feature_data['importance_score'],
                    mode='lines+markers',
                    name=feature,
                    line=dict(width=2),
                    marker=dict(size=6)
                ))

        # Customize layout
        fig.update_layout(
            title='Interactive Feature Importance Trends',
            xaxis_title='Date',
            yaxis_title='Importance Score',
            hovermode='x unified',
            showlegend=True,
            height=600
        )

        return fig

    def create_interactive_heatmap(self,
                                 results: Dict[ImportanceMethod, ImportanceResult]) -> Optional[go.Figure]:
        """
        Create interactive heatmap using Plotly.

        Args:
            results: Dictionary of importance results by method

        Returns:
            Plotly figure or None if Plotly not available
        """
        if not PLOTLY_AVAILABLE or not results:
            return None

        # Prepare data
        all_features = set()
        for result in results.values():
            all_features.update(score.feature_name for score in result.scores)

        all_features = sorted(all_features)
        method_names = [method.value for method in results.keys()]

        # Create importance matrix
        importance_matrix = []
        for method, result in results.items():
            method_scores = []
            for feature in all_features:
                feature_scores = [s.score for s in result.scores if s.feature_name == feature]
                method_scores.append(np.mean(feature_scores) if feature_scores else 0.0)
            importance_matrix.append(method_scores)

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=importance_matrix,
            x=all_features,
            y=method_names,
            colorscale='YlOrRd',
            hoverongaps=False,
            hoverinfo='text',
            text=[[f'{val:.3f}' for val in row] for row in importance_matrix]
        ))

        # Customize layout
        fig.update_layout(
            title='Interactive Feature Importance Heatmap',
            xaxis_title='Features',
            yaxis_title='Methods',
            height=max(400, len(method_names) * 50)
        )

        return fig

    def create_interactive_dashboard(self,
                                   feature_names: List[str],
                                   lookback_days: int = 30) -> Optional[go.Figure]:
        """
        Create interactive dashboard using Plotly subplots.

        Args:
            feature_names: Features to include
            lookback_days: Number of days to look back

        Returns:
            Plotly figure or None if Plotly not available
        """
        if not PLOTLY_AVAILABLE or self.tracker is None:
            return None

        # Get data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        history_df = self.tracker.get_importance_history(
            feature_names=feature_names,
            start_time=start_date,
            end_time=end_date
        )

        if history_df.empty:
            return None

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Top Features', 'Importance Trend', 'Method Distribution', 'Feature Volatility'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "pie"}, {"type": "bar"}]]
        )

        # Subplot 1: Top features
        if not history_df.empty:
            top_features = history_df.groupby('feature_name')['importance_score'].mean().nlargest(10)
            fig.add_trace(go.Bar(
                x=top_features.values,
                y=top_features.index,
                orientation='h',
                name='Top Features'
            ), row=1, col=1)

        # Subplot 2: Trend over time
        if not history_df.empty:
            daily_avg = history_df.groupby(history_df['timestamp'].dt.date)['importance_score'].mean()
            fig.add_trace(go.Scatter(
                x=daily_avg.index,
                y=daily_avg.values,
                mode='lines+markers',
                name='Average Trend'
            ), row=1, col=2)

        # Subplot 3: Method usage
        if not history_df.empty:
            method_counts = history_df['method'].value_counts()
            fig.add_trace(go.Pie(
                labels=method_counts.index,
                values=method_counts.values,
                name="Methods"
            ), row=2, col=1)

        # Subplot 4: Feature volatility
        if not history_df.empty:
            volatility = history_df.groupby('feature_name')['importance_score'].std() / history_df.groupby('feature_name')['importance_score'].mean()
            volatility = volatility.dropna().sort_values(ascending=False).head(10)
            fig.add_trace(go.Bar(
                x=volatility.index,
                y=volatility.values,
                name='Volatility'
            ), row=2, col=2)

        # Update layout
        fig.update_layout(
            title_text=f'Interactive Importance Dashboard - Last {lookback_days} Days',
            showlegend=False,
            height=800
        )

        return fig

    def save_interactive_plot(self,
                            fig: go.Figure,
                            filename: str,
                            format: str = 'html'):
        """
        Save interactive plot to file.

        Args:
            fig: Plotly figure
            filename: Output filename
            format: Output format ('html', 'png', 'pdf')
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available - cannot save interactive plots")
            return

        output_path = Path(filename)

        if format.lower() == 'html':
            fig.write_html(str(output_path))
        elif format.lower() == 'png':
            fig.write_image(str(output_path))
        elif format.lower() == 'pdf':
            fig.write_image(str(output_path))
        else:
            logger.warning(f"Unsupported format: {format}")

        logger.info(f"Interactive plot saved to {output_path}")


def create_importance_report(tracker: ImportanceTracker,
                           feature_names: List[str],
                           output_dir: str = "importance_reports",
                           lookback_days: int = 30,
                           create_interactive: bool = True) -> Dict[str, str]:
    """
    Create comprehensive importance report with multiple visualizations.

    Args:
        tracker: ImportanceTracker instance
        feature_names: Features to include in report
        output_dir: Directory to save report files
        lookback_days: Number of days to look back
        create_interactive: Whether to create interactive plots

    Returns:
        Dictionary mapping report types to file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Initialize visualizers
    static_viz = ImportanceVisualizer(tracker=tracker, output_dir=output_path)
    interactive_viz = InteractiveImportanceVisualizer(tracker=tracker) if create_interactive else None

    report_files = {}

    # Get data for analysis
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)

    # Generate static visualizations
    logger.info("Generating static visualizations...")

    # 1. Summary dashboard
    dashboard_path = output_path / "importance_dashboard.png"
    static_viz.create_summary_dashboard(
        feature_names=feature_names,
        lookback_days=lookback_days,
        save_path=str(dashboard_path)
    )
    report_files['dashboard'] = str(dashboard_path)

    # 2. Temporal trends
    trends_path = output_path / "importance_trends.png"
    static_viz.create_temporal_trend_plot(
        feature_names=feature_names,
        start_date=start_date,
        end_date=end_date,
        save_path=str(trends_path)
    )
    report_files['trends'] = str(trends_path)

    # 3. Recent drift alerts
    recent_alerts = list(tracker.alerts)[-20:]  # Last 20 alerts
    if recent_alerts:
        alerts_path = output_path / "drift_alerts.png"
        static_viz.create_drift_alert_plot(
            alerts=recent_alerts,
            save_path=str(alerts_path)
        )
        report_files['alerts'] = str(alerts_path)

    # 4. Feature correlation matrix
    if len(feature_names) >= 2:
        corr_path = output_path / "feature_correlation.png"
        static_viz.create_correlation_matrix(
            feature_names=feature_names,
            start_date=start_date,
            end_date=end_date,
            save_path=str(corr_path)
        )
        report_files['correlation'] = str(corr_path)

    # Generate interactive visualizations
    if interactive_viz and PLOTLY_AVAILABLE:
        logger.info("Generating interactive visualizations...")

        # Interactive dashboard
        interactive_fig = interactive_viz.create_interactive_dashboard(
            feature_names=feature_names,
            lookback_days=lookback_days
        )
        if interactive_fig:
            interactive_path = output_path / "interactive_dashboard.html"
            interactive_viz.save_interactive_plot(interactive_fig, str(interactive_path))
            report_files['interactive_dashboard'] = str(interactive_path)

        # Interactive trend plot
        trend_fig = interactive_viz.create_interactive_trend_plot(
            feature_names=feature_names,
            start_date=start_date,
            end_date=end_date
        )
        if trend_fig:
            interactive_trend_path = output_path / "interactive_trends.html"
            interactive_viz.save_interactive_plot(trend_fig, str(interactive_trend_path))
            report_files['interactive_trends'] = str(interactive_trend_path)

    # Export summary statistics
    summary = tracker.get_importance_statistics()
    summary_path = output_path / "importance_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    report_files['summary'] = str(summary_path)

    logger.info(f"Importance report created in {output_path}")
    return report_files