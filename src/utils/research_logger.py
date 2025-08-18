#!/usr/bin/env python3
"""
Research Logger for Structured Training Logs
===========================================

Provides minimal console logging and detailed file logging for research tracking.
"""

import os
import logging
import sys
from datetime import datetime
from typing import Dict, Any, Optional
from tqdm import tqdm
import pandas as pd
from .model_analyzer import ModelAnalyzer


class ResearchLogger:
    """
    Enhanced logger with minimal console output and detailed file logging.
    """
    
    def __init__(self, config: Dict[str, Any], iteration_dir: str, use_progress_bar: bool = False):
        self.config = config
        self.logging_config = config.get('logging', {})
        self.iteration_dir = iteration_dir
        self.use_progress_bar = use_progress_bar
        
        # Set up loggers
        self.console_logger = self._setup_console_logger()
        self.file_logger = self._setup_file_logger()
        
        # Progress bar
        self.progress_bar = None
        self.episode_progress = None
        
        # Episode-specific tracking
        self.current_episode_metrics = {}
        self.episode_start_time = None
        
    def _setup_console_logger(self) -> logging.Logger:
        """Set up console logger with minimal output."""
        logger = logging.getLogger('research.console')
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        if not self.use_progress_bar:
            # Create console handler with minimal format
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            
            # Use simple format for console
            console_format = logging.Formatter('%(message)s')
            console_handler.setFormatter(console_format)
            logger.addHandler(console_handler)
        
        logger.propagate = False
        return logger
    
    def _setup_file_logger(self) -> logging.Logger:
        """Set up detailed file logger for iteration tracking."""
        logger = logging.getLogger('research.detailed')
        logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create file handler for detailed logs with UTF-8 encoding
        log_file = os.path.join(self.iteration_dir, 'training_detailed.log')
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Use detailed format for file logging
        detailed_format = self.logging_config.get('detailed_format', 
                                                '%(asctime)s - %(levelname)s - %(message)s')
        file_formatter = logging.Formatter(detailed_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        logger.propagate = False
        return logger
    
    def start_training(self, total_episodes: int, symbols: list):
        """Initialize training logging."""
        if self.use_progress_bar:
            self.progress_bar = tqdm(
                total=total_episodes,
                desc="Training Progress",
                unit="episode",
                position=0,
                leave=True
            )
        
        # Log training start
        start_msg = f"Training Started: {total_episodes} episodes across {len(symbols)} symbols"
        self._log_both(start_msg)
        
        # Detailed file logging
        self.file_logger.info("=" * 80)
        self.file_logger.info("TRAINING SESSION STARTED")
        self.file_logger.info("=" * 80)
        self.file_logger.info(f"Total Episodes: {total_episodes}")
        self.file_logger.info(f"Symbols: {symbols}")
        self.file_logger.info(f"Configuration Hash: {getattr(self, 'config_hash', 'unknown')}")
        self.file_logger.info(f"Iteration Directory: {self.iteration_dir}")
        self.file_logger.info("=" * 80)
    
    def start_episode(self, episode: int, symbol: str, initial_capital: float):
        """Start episode logging."""
        self.episode_start_time = datetime.now()
        self.current_episode_metrics = {
            'episode': episode,
            'symbol': symbol,
            'initial_capital': initial_capital,
            'steps': 0,
            'trades': 0,
            'win_rate': 0.0,
            'current_capital': initial_capital,
            'total_pnl': 0.0
        }
        
        # Update progress bar
        if self.progress_bar:
            self.progress_bar.set_description(f"Episode {episode} | {symbol}")
        
        # Detailed file logging
        self.file_logger.info(f"\n--- EPISODE {episode} START ---")
        self.file_logger.info(f"Symbol: {symbol}")
        self.file_logger.info(f"Initial Capital: ₹{initial_capital:,.2f}")
        self.file_logger.info(f"Start Time: {self.episode_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def log_step(self, step: int, step_data: Dict[str, Any]):
        """Log a single training step with minimal console output."""
        # Extract data from step_data
        datetime_str = step_data.get('datetime', 'N/A')
        instrument = step_data.get('instrument', 'N/A')
        action_name = step_data.get('action_name', 'UNKNOWN')
        quantity = step_data.get('quantity', 0)
        win_rate = step_data.get('episode_win_rate', 0) * 100  # Convert to percentage
        initial_capital = step_data.get('initial_capital', 0)
        current_capital = step_data.get('current_capital', 0)
        total_pnl = current_capital - initial_capital
        entry_price = step_data.get('entry_price', 'N/A')
        sl_price = step_data.get('sl_price', 'N/A')
        target_price = step_data.get('target_price', 'N/A')
        
        # Update episode metrics
        self.current_episode_metrics.update({
            'steps': step,
            'current_capital': current_capital,
            'total_pnl': total_pnl,
            'win_rate': win_rate
        })
        
        # Minimal console logging (only show key steps)
        if not self.use_progress_bar:
            # Build action string with probabilities
            if action_name == "HOLD":
                action_str = action_name
            else:
                action_str = f"{action_name}-{int(quantity)}" if quantity > 0 else action_name
            
            # Add action probabilities if available
            action_prob_str = ""
            prob_key_map = {
                'BUY_LONG': 'prob_BUY',
                'SELL_SHORT': 'prob_SELL', 
                'CLOSE_LONG': 'prob_CLOSE_L',
                'CLOSE_SHORT': 'prob_CLOSE_S',
                'HOLD': 'prob_HOLD'
            }
            
            if action_name in prob_key_map and prob_key_map[action_name] in step_data:
                action_prob_str = f" ({step_data[prob_key_map[action_name]]})"
            
            action_display = f"{action_str}{action_prob_str}"
            
            # Format prices for display
            entry_str = f"Rs.{entry_price:,.0f}" if isinstance(entry_price, (int, float)) else str(entry_price)
            sl_str = f"Rs.{sl_price:,.0f}" if isinstance(sl_price, (int, float)) else str(sl_price)
            target_str = f"Rs.{target_price:,.0f}" if isinstance(target_price, (int, float)) else str(target_price)
            
            # Include exit reason in console output if available
            exit_reason = step_data.get('exit_reason', '')
            exit_str = f" | Exit: {exit_reason}" if exit_reason else ""
            
            minimal_log = (f"{step:3d} | {datetime_str} | {instrument} | {action_display:18s} | "
                         f"WR: {win_rate:5.1f}% | Rs.{initial_capital:7,.0f} -> Rs.{current_capital:7,.0f} | "
                         f"P&L: Rs.{total_pnl:7,.0f} | {entry_str} | {sl_str} | {target_str}{exit_str}")
            
            self.console_logger.info(minimal_log)
        
        # Single-line detailed file logging with | separators
        reward_str = f" | Reward: {step_data.get('reward', 0):.4f}" if 'reward' in step_data else ""
        exit_str = f" | Exit: {step_data.get('exit_reason', '')}" if step_data.get('exit_reason') else ""
        
        detailed_log = (f"STEP {step:4d} | {datetime_str} | {instrument} | {action_name} | "
                       f"Qty: {quantity} | Capital: Rs{initial_capital:,.0f} -> Rs{current_capital:,.0f} | "
                       f"P&L: Rs{total_pnl:,.0f} | WR: {win_rate:.1f}% | "
                       f"Entry: {entry_price} | SL: {sl_price} | Target: {target_price}{reward_str}{exit_str}")
        
        self.file_logger.debug(detailed_log)
        
        # Update progress bar
        if self.progress_bar:
            progress_metrics = f"WR:{win_rate:.1f}% | ₹{current_capital:,.0f} | P&L:₹{total_pnl:,.0f}"
            self.progress_bar.set_postfix_str(progress_metrics)
    
    def end_episode(self, episode_metrics: Dict[str, Any]):
        """End episode logging with summary."""
        end_time = datetime.now()
        duration = (end_time - self.episode_start_time).total_seconds() if self.episode_start_time else 0
        
        # Update episode metrics
        self.current_episode_metrics.update(episode_metrics)
        final_metrics = self.current_episode_metrics
        
        # Console summary
        if not self.use_progress_bar:
            episode_summary = (f"Episode {final_metrics['episode']} Complete | {final_metrics['symbol']} | "
                             f"WR: {final_metrics['win_rate']:.1f}% | Rs.{final_metrics['current_capital']:,.0f} | "
                             f"P&L: Rs.{final_metrics['total_pnl']:,.0f} | {duration:.1f}s")
            self.console_logger.info(episode_summary)
        
        # Update progress bar
        if self.progress_bar:
            self.progress_bar.update(1)
        
        # Detailed file logging
        self.file_logger.info(f"--- EPISODE {final_metrics['episode']} SUMMARY ---")
        self.file_logger.info(f"Duration: {duration:.2f} seconds")
        self.file_logger.info(f"Total Steps: {final_metrics['steps']}")
        self.file_logger.info(f"Final Capital: ₹{final_metrics['current_capital']:,.2f}")
        self.file_logger.info(f"Total P&L: ₹{final_metrics['total_pnl']:,.2f}")
        self.file_logger.info(f"Episode Win Rate: {final_metrics['win_rate']:.2f}%")
        self.file_logger.info(f"Total Trades: {final_metrics.get('trades', 0)}")
        
        # Save episode metrics to CSV for analysis
        self._save_episode_metrics(final_metrics)
    
    def end_training(self, training_summary: Dict[str, Any], model=None):
        """End training logging with final summary and model analysis."""
        if self.progress_bar:
            self.progress_bar.close()
        
        # Console summary
        final_summary = (f"Training Complete | Episodes: {training_summary.get('total_episodes', 0)} | "
                        f"Final Capital: Rs.{training_summary.get('final_capital', 0):,.0f} | "
                        f"Total Return: {training_summary.get('total_return_pct', 0):.2f}%")
        self._log_both(final_summary)
        
        # Model analysis
        if model is not None:
            self._log_model_analysis(model)
        
        # Detailed file logging
        self.file_logger.info("=" * 80)
        self.file_logger.info("TRAINING SESSION COMPLETED")
        self.file_logger.info("=" * 80)
        for key, value in training_summary.items():
            self.file_logger.info(f"{key}: {value}")
        self.file_logger.info("=" * 80)
    
    def _log_model_analysis(self, model):
        """Log comprehensive model analysis."""
        try:
            analysis = ModelAnalyzer.analyze_model(model)
            
            # Concise console summary
            concise_summary = ModelAnalyzer.generate_concise_summary(analysis)
            self._log_both(f"[MODEL] {concise_summary}")
            
            # Detailed file logging
            detailed_lines = ModelAnalyzer.generate_detailed_summary(analysis)
            self.file_logger.info("")
            for line in detailed_lines:
                self.file_logger.info(line)
            
        except Exception as e:
            self.file_logger.warning(f"Could not analyze model architecture: {e}")
    
    def _save_episode_metrics(self, metrics: Dict[str, Any]):
        """Save episode metrics to CSV for analysis."""
        csv_file = os.path.join(self.iteration_dir, 'episode_metrics.csv')
        
        # Create DataFrame from metrics
        df_new = pd.DataFrame([metrics])
        
        # Append to existing CSV or create new one
        if os.path.exists(csv_file):
            df_existing = pd.read_csv(csv_file)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new
        
        df_combined.to_csv(csv_file, index=False)
    
    def _log_both(self, message: str):
        """Log to both console and file."""
        # Remove emojis for console logging to avoid encoding issues
        console_message = message
        if not self.use_progress_bar:
            # Remove common emojis that might cause encoding issues
            emoji_chars = ['🎯', '📊', '💰', '📈', '📉', '🛑', '✅', '🔄', '⚠️', '❌', '🟢', '🔴', '🟥', '🟩', '🟦', '⬜', '⬛', '🔶', '🔷', '🔸', '🔹']
            for emoji in emoji_chars:
                console_message = console_message.replace(emoji, '')
            # Also replace the chart emoji
            console_message = console_message.replace('📈', '').replace('📉', '').replace('🎯', '').replace('📊', '')
            console_message = console_message.replace('💰', '').replace('🛑', '').replace('✅', '').replace('🔄', '')
            console_message = console_message.replace('⚠️', '').replace('❌', '').replace('🟢', '').replace('🔴', '')
            self.console_logger.info(console_message.strip())
        self.file_logger.info(message)
    
    def log_info(self, message: str):
        """Log informational message."""
        self._log_both(message)
    
    def log_warning(self, message: str):
        """Log warning message."""
        self._log_both(f"⚠️ {message}")
        
    def log_error(self, message: str):
        """Log error message."""
        self._log_both(f"❌ {message}")