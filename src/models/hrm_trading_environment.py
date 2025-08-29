import torch
import numpy as np
import gymnasium as gym
from typing import Dict, Tuple, Optional, List
import logging
import yaml
from pathlib import Path

from src.env.environment import TradingEnv
from src.env.trading_mode import TradingMode
from src.models.hrm import HRMTradingAgent, HRMCarry
from src.utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class HRMTradingEnvironment(TradingEnv):
    """
    HRM-enhanced trading environment that integrates hierarchical reasoning
    with the existing trading infrastructure.
    """
    
    def __init__(self, 
                 data_loader=None, 
                 symbol=None, 
                 initial_capital=None,
                 hrm_config_path="config/hrm_config.yaml",
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 **kwargs):
        
        # Load HRM configuration
        self.hrm_config = self._load_hrm_config(hrm_config_path)
        self.device = torch.device(device)
        
        # Initialize base trading environment
        super().__init__(
            data_loader=data_loader,
            symbol=symbol,
            initial_capital=initial_capital,
            **kwargs
        )
        
        # HRM-specific initialization
        self.hrm_agent = None
        self.current_carry = None
        self.deep_supervision_active = False
        self.segment_count = 0
        self.max_segments = self.hrm_config['model_architecture']['deep_supervision_segments']
        
        # Performance tracking for HRM
        self.hierarchical_metrics = {
            'strategic_decisions': [],
            'tactical_decisions': [],
            'regime_classifications': [],
            'act_computations': [],
            'reward_history': []
        }
        
        # Strategic update frequency
        self.strategic_update_frequency = self.hrm_config['environment']['strategic_update_frequency']
        self.steps_since_strategic_update = 0
        
        # Market context for embeddings
        from src.utils.instruments_loader import get_instruments_loader
        self.instruments_loader = get_instruments_loader()
        self.current_instrument_id = None
        self.current_timeframe_id = None
        self._resolve_market_context()
        
    def _load_hrm_config(self, config_path: str) -> Dict:
        """Load HRM configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Loaded HRM configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load HRM configuration: {e}")
            # Return default configuration
            return self._get_default_hrm_config()
    
    def _get_default_hrm_config(self) -> Dict:
        """Return default HRM configuration"""
        return {
            'model_architecture': {
                'hidden_dim': 256,
                'H_cycles': 2,
                'L_cycles': 5,
                'H_layers': 4,
                'L_layers': 3,
                'num_heads': 8,
                'halt_max_steps': 8,
                'halt_exploration_prob': 0.1,
                'deep_supervision_segments': 4,
                'dropout': 0.1
            },
            'environment': {
                'strategic_update_frequency': 10,
                'enable_hierarchical_observations': True
            }
        }
    
    def reset(self, performance_metrics: Optional[Dict] = None) -> np.ndarray:
        """Reset environment and initialize HRM agent"""
        
        # Reset base environment
        observation = super().reset(performance_metrics)
        
        # Initialize HRM agent if not already done
        if self.hrm_agent is None:
            self._initialize_hrm_agent()
        
        # Create fresh carry state
        self.current_carry = self.hrm_agent.create_initial_carry(batch_size=1)
        
        # Reset HRM-specific tracking
        self.deep_supervision_active = False
        self.segment_count = 0
        self.steps_since_strategic_update = 0
        
        # Reset hierarchical metrics
        for key in self.hierarchical_metrics:
            self.hierarchical_metrics[key] = []
            
        logger.info("HRM Trading Environment reset completed")
        
        return observation
    
    def _resolve_market_context(self):
        """Resolve instrument and timeframe IDs from current symbol and config"""
        try:
            # Extract timeframe from environment config or data
            timeframe = getattr(self, 'timeframe', None)
            if not timeframe and hasattr(self, 'data') and hasattr(self.data, 'columns'):
                # Try to extract from data structure or default to 5min
                timeframe = "5"
            
            # Get IDs from instruments loader
            symbol = getattr(self, 'symbol', None)
            if symbol and timeframe:
                valid, instrument_id, timeframe_id = self.instruments_loader.validate_symbol_timeframe(symbol, timeframe)
                if valid:
                    self.current_instrument_id = instrument_id
                    self.current_timeframe_id = timeframe_id
                    logger.info(f"Resolved market context: {symbol} (ID: {instrument_id}), {timeframe} (ID: {timeframe_id})")
                else:
                    logger.warning(f"Could not resolve market context for {symbol}_{timeframe}")
                    # Use default IDs (0, 0) for unknown instruments/timeframes
                    self.current_instrument_id = 0
                    self.current_timeframe_id = 0
            else:
                logger.warning("Symbol or timeframe not available, using default context")
                self.current_instrument_id = 0
                self.current_timeframe_id = 0
                
        except Exception as e:
            logger.error(f"Error resolving market context: {e}")
            self.current_instrument_id = 0
            self.current_timeframe_id = 0
    
    def _initialize_hrm_agent(self):
        """Initialize HRM trading agent based on observation space"""
        
        if self.observation_space is None:
            raise ValueError("Observation space not initialized. Call reset() on base environment first.")
        
        # Get HRM model configuration
        model_config = self.hrm_config['model_architecture']
        
        # Calculate feature dimension from observation space
        # Observation space is flattened: [lookback_window * features_per_step + account_features]
        hierarchical_config = self.hrm_config['hierarchical_processing']
        
        # Get the feature dimension per timestep (excluding account state features)
        account_features = 6  # From observation_handler.py
        total_obs_dim = self.observation_space.shape[0]
        market_features_dim = total_obs_dim - account_features
        
        # Get lookback window from observation handler
        lookback_window = getattr(self.observation_handler, 'lookback_window', 50)
        feature_dim = market_features_dim // lookback_window
        
        # Initialize HRM agent with dynamic feature detection
        self.hrm_agent = HRMTradingAgent(
            feature_dim=feature_dim,
            h_lookback=hierarchical_config['high_level_lookback'],
            l_lookback=hierarchical_config['low_level_lookback'],
            hidden_dim=model_config['hidden_dim'],
            H_layers=model_config['H_layers'],
            L_layers=model_config['L_layers'],
            num_heads=model_config['num_heads'],
            H_cycles=model_config['H_cycles'],
            L_cycles=model_config['L_cycles'],
            halt_max_steps=model_config['halt_max_steps'],
            halt_exploration_prob=model_config['halt_exploration_prob'],
            dropout=model_config['dropout'],
            device=str(self.device)
        )
        
        logger.info(f"HRM Trading Agent initialized with {sum(p.numel() for p in self.hrm_agent.parameters())} parameters")
    
    def step(self, raw_observation: np.ndarray = None) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Enhanced step function with HRM reasoning
        
        Args:
            raw_observation: If provided, use this observation instead of generating one
        """
        
        # Get current observation if not provided
        if raw_observation is None:
            # Use hierarchical observation for HRM agent
            hierarchical_obs = self.observation_handler.get_hierarchical_observation(
                self.data, self.current_step, self.engine
            )
            # For now, use the high-level observation as the main observation
            # This will be further processed by the HRM agent for hierarchical reasoning
            raw_observation = hierarchical_obs['high_level']
        
        # Convert to torch tensor
        observation_tensor = torch.FloatTensor(raw_observation).unsqueeze(0).to(self.device)
        
        # Prepare market context tensors
        instrument_tensor = torch.tensor([self.current_instrument_id], dtype=torch.long, device=self.device)
        timeframe_tensor = torch.tensor([self.current_timeframe_id], dtype=torch.long, device=self.device)
        
        # Deep supervision loop - multiple forward passes for enhanced learning
        segment_rewards = []
        segment_actions = []
        
        for segment in range(self.max_segments):
            # Forward pass through HRM with market context
            self.current_carry, outputs = self.hrm_agent.forward(
                self.current_carry, 
                observation_tensor, 
                training=self.mode == TradingMode.TRAINING,
                instrument_id=instrument_tensor,
                timeframe_id=timeframe_tensor
            )
            
            # Extract trading decision from HRM outputs
            trading_decision = self.hrm_agent.extract_trading_decision(outputs)
            action = (trading_decision['action_idx'], trading_decision['quantity'])
            segment_actions.append(action)
            
            # Execute trade in environment
            next_observation, reward, done, info = super().step(action)
            segment_rewards.append(reward)
            
            # Update performance metrics in HRM agent
            if hasattr(self, 'reward_calculator'):
                total_return = (self.engine.get_account_state()['capital'] - self.initial_capital) / self.initial_capital
                sharpe_ratio = self._calculate_sharpe_ratio()
                max_drawdown = self.termination_manager.get_current_drawdown()
                
                self.hrm_agent.update_performance_metrics(
                    self.current_carry, reward, total_return, sharpe_ratio, max_drawdown
                )
            
            # Track hierarchical metrics
            self._track_hierarchical_metrics(outputs, action, reward)
            
            # Check if computation should halt
            if self.current_carry.halted.all():
                break
        
        # Aggregate results from deep supervision
        final_observation = next_observation
        final_reward = self._aggregate_segment_rewards(segment_rewards)
        final_done = done
        
        # Enhanced info with HRM insights
        final_info = info.copy() if info else {}
        final_info.update(self._get_hrm_insights(outputs))
        final_info['hrm_segments_used'] = segment + 1
        final_info['segment_rewards'] = segment_rewards
        
        self.steps_since_strategic_update += 1
        
        return final_observation, final_reward, final_done, final_info
    
    def _aggregate_segment_rewards(self, segment_rewards: List[float]) -> float:
        """Aggregate rewards from multiple supervision segments"""
        if not segment_rewards:
            return 0.0
        
        # Use weighted average based on segment weights
        segment_weights = self.hrm_config['model_architecture']['segment_weights']
        
        # Ensure we have enough weights
        if len(segment_weights) < len(segment_rewards):
            # Extend weights with equal weighting for remaining segments
            remaining_weight = (1.0 - sum(segment_weights)) / max(1, len(segment_rewards) - len(segment_weights))
            segment_weights.extend([remaining_weight] * (len(segment_rewards) - len(segment_weights)))
        
        # Calculate weighted average
        weighted_reward = sum(w * r for w, r in zip(segment_weights[:len(segment_rewards)], segment_rewards))
        return weighted_reward
    
    def _track_hierarchical_metrics(self, outputs: Dict, action: Tuple, reward: float):
        """Track HRM-specific performance metrics"""
        
        # Extract and store strategic decisions
        if 'regime_probabilities' in outputs:
            regime_probs = outputs['regime_probabilities'].detach().cpu().numpy()[0]
            self.hierarchical_metrics['regime_classifications'].append(regime_probs)
        
        # Track strategic vs tactical decision quality
        if 'signal_strength' in outputs:
            signal_strength = outputs['signal_strength'].detach().cpu().numpy()[0]
            self.hierarchical_metrics['strategic_decisions'].append(signal_strength)
        
        if 'confidence' in outputs:
            confidence = outputs['confidence'].detach().cpu().numpy()[0]
            self.hierarchical_metrics['tactical_decisions'].append(confidence)
        
        # Track ACT efficiency
        if 'halt_logits' in outputs and 'continue_logits' in outputs:
            halt_prob = torch.sigmoid(outputs['halt_logits'] - outputs['continue_logits']).detach().cpu().numpy()[0]
            self.hierarchical_metrics['act_computations'].append(halt_prob)
        
        # Store reward for analysis
        self.hierarchical_metrics['reward_history'].append(reward)
    
    def _get_hrm_insights(self, outputs: Dict) -> Dict:
        """Extract interpretable insights from HRM outputs"""
        
        insights = {}
        
        try:
            # Market regime analysis
            if 'regime_probabilities' in outputs:
                regime_analysis = self.hrm_agent.high_level_module.interpret_regime(outputs['regime_probabilities'])
                insights['market_regime'] = regime_analysis['regime_enum'].name
                insights['regime_confidence'] = regime_analysis['regime_confidence']
            
            # Risk assessment
            risk_params = outputs.get('risk_parameters', torch.zeros(1, 4))
            insights['risk_assessment'] = {
                'risk_multiplier': risk_params[0, 0].item(),
                'position_size_factor': risk_params[0, 1].item(),
                'max_exposure': risk_params[0, 2].item(),
                'volatility_threshold': risk_params[0, 3].item()
            }
            
            # Computation efficiency
            if 'halt_logits' in outputs and 'continue_logits' in outputs:
                halt_prob = torch.sigmoid(outputs['halt_logits'] - outputs['continue_logits'])
                insights['computation_efficiency'] = halt_prob.item()
            
        except Exception as e:
            logger.warning(f"Failed to extract HRM insights: {e}")
            insights['extraction_error'] = str(e)
        
        return insights
    
    def _calculate_sharpe_ratio(self, lookback: int = 100) -> float:
        """Calculate Sharpe ratio from recent performance"""
        
        if len(self.hierarchical_metrics['reward_history']) < 10:
            return 0.0
        
        recent_rewards = self.hierarchical_metrics['reward_history'][-lookback:]
        
        if not recent_rewards:
            return 0.0
        
        mean_return = np.mean(recent_rewards)
        std_return = np.std(recent_rewards)
        
        if std_return == 0:
            return 0.0
        
        return mean_return / std_return
    
    def get_hierarchical_performance_metrics(self) -> Dict:
        """Get comprehensive HRM performance metrics"""
        
        metrics = {}
        
        try:
            # Strategic decision quality
            if self.hierarchical_metrics['strategic_decisions']:
                strategic_decisions = np.array(self.hierarchical_metrics['strategic_decisions'])
                metrics['strategic_performance'] = {
                    'mean_signal_strength': np.mean(strategic_decisions),
                    'signal_consistency': 1 - np.std(strategic_decisions),
                    'decisions_count': len(strategic_decisions)
                }
            
            # Tactical decision quality
            if self.hierarchical_metrics['tactical_decisions']:
                tactical_decisions = np.array(self.hierarchical_metrics['tactical_decisions'])
                metrics['tactical_performance'] = {
                    'mean_confidence': np.mean(tactical_decisions),
                    'confidence_consistency': 1 - np.std(tactical_decisions),
                    'decisions_count': len(tactical_decisions)
                }
            
            # Regime classification accuracy
            if self.hierarchical_metrics['regime_classifications']:
                regime_data = np.array(self.hierarchical_metrics['regime_classifications'])
                metrics['regime_analysis'] = {
                    'dominant_regimes': self._analyze_regime_distribution(regime_data),
                    'regime_certainty': np.mean(np.max(regime_data, axis=1)),
                    'regime_diversity': self._calculate_regime_diversity(regime_data)
                }
            
            # ACT efficiency
            if self.hierarchical_metrics['act_computations']:
                act_data = np.array(self.hierarchical_metrics['act_computations'])
                metrics['act_performance'] = {
                    'mean_halt_probability': np.mean(act_data),
                    'computation_efficiency': 1 - np.mean(act_data),  # Lower halt prob = more efficient
                    'halt_consistency': 1 - np.std(act_data)
                }
            
            # Overall coordination
            metrics['hierarchical_coordination'] = self._assess_hierarchical_coordination()
            
        except Exception as e:
            logger.error(f"Error calculating hierarchical metrics: {e}")
            metrics['calculation_error'] = str(e)
        
        return metrics
    
    def _analyze_regime_distribution(self, regime_data: np.ndarray) -> Dict:
        """Analyze distribution of market regimes"""
        
        # Get dominant regime for each timestep
        dominant_regimes = np.argmax(regime_data, axis=1)
        
        # Calculate distribution
        unique, counts = np.unique(dominant_regimes, return_counts=True)
        
        regime_names = ['TRENDING_UP', 'TRENDING_DOWN', 'RANGING', 'HIGH_VOLATILITY', 'LOW_VOLATILITY']
        
        distribution = {}
        for regime_id, count in zip(unique, counts):
            if regime_id < len(regime_names):
                distribution[regime_names[regime_id]] = count / len(dominant_regimes)
        
        return distribution
    
    def _calculate_regime_diversity(self, regime_data: np.ndarray) -> float:
        """Calculate how diverse the regime classifications are"""
        
        # Shannon entropy of regime probabilities
        mean_probs = np.mean(regime_data, axis=0)
        mean_probs = mean_probs[mean_probs > 0]  # Remove zero probabilities
        
        if len(mean_probs) == 0:
            return 0.0
        
        entropy = -np.sum(mean_probs * np.log(mean_probs))
        max_entropy = np.log(len(mean_probs))
        
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _assess_hierarchical_coordination(self) -> Dict:
        """Assess coordination between H and L modules"""
        
        coordination_metrics = {}
        
        try:
            strategic_decisions = self.hierarchical_metrics['strategic_decisions']
            tactical_decisions = self.hierarchical_metrics['tactical_decisions']
            rewards = self.hierarchical_metrics['reward_history']
            
            if len(strategic_decisions) > 0 and len(tactical_decisions) > 0:
                # Correlation between strategic and tactical decisions
                min_len = min(len(strategic_decisions), len(tactical_decisions))
                if min_len > 10:
                    strategic_subset = strategic_decisions[-min_len:]
                    tactical_subset = tactical_decisions[-min_len:]
                    
                    correlation = np.corrcoef(strategic_subset, tactical_subset)[0, 1]
                    coordination_metrics['strategic_tactical_correlation'] = correlation if not np.isnan(correlation) else 0.0
            
            # Consistency of decision quality with rewards
            if len(rewards) > 10:
                recent_rewards = rewards[-100:]  # Last 100 rewards
                reward_trend = np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0]
                coordination_metrics['performance_trend'] = reward_trend
            
        except Exception as e:
            logger.warning(f"Error assessing hierarchical coordination: {e}")
            coordination_metrics['assessment_error'] = str(e)
        
        return coordination_metrics
    
    def save_hrm_state(self, filepath: str):
        """Save HRM agent state"""
        
        if self.hrm_agent is None:
            logger.warning("No HRM agent to save")
            return
        
        try:
            torch.save({
                'model_state_dict': self.hrm_agent.state_dict(),
                'config': self.hrm_config,
                'hierarchical_metrics': self.hierarchical_metrics
            }, filepath)
            logger.info(f"HRM state saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save HRM state: {e}")
    
    def load_hrm_state(self, filepath: str):
        """Load HRM agent state"""
        
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            if self.hrm_agent is None:
                self._initialize_hrm_agent()
            
            self.hrm_agent.load_state_dict(checkpoint['model_state_dict'])
            self.hierarchical_metrics = checkpoint.get('hierarchical_metrics', self.hierarchical_metrics)
            
            logger.info(f"HRM state loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load HRM state: {e}")


class HRMTradingWrapper:
    """
    Wrapper class to integrate HRM with existing training loops
    """
    
    def __init__(self, 
                 base_env: TradingEnv,
                 hrm_config_path: str = "config/hrm_config.yaml",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        
        self.base_env = base_env
        self.device = torch.device(device)
        
        # Convert base environment to HRM environment
        self.hrm_env = HRMTradingEnvironment(
            data_loader=base_env.data_loader,
            symbol=base_env.symbol,
            initial_capital=base_env.initial_capital,
            hrm_config_path=hrm_config_path,
            device=device,
            lookback_window=base_env.lookback_window,
            reward_function=base_env.reward_function,
            episode_length=base_env.episode_length,
            mode=base_env.mode
        )
        
        # Copy environment state
        self.hrm_env.data = base_env.data
        self.hrm_env.current_step = base_env.current_step
        
    def step(self, observation: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Step function that takes observation and returns environment step results"""
        return self.hrm_env.step(observation)
    
    def reset(self, **kwargs) -> np.ndarray:
        """Reset the HRM environment"""
        return self.hrm_env.reset(**kwargs)
    
    def get_hrm_insights(self) -> Dict:
        """Get HRM-specific insights and metrics"""
        return self.hrm_env.get_hierarchical_performance_metrics()
    
    @property
    def observation_space(self):
        return self.hrm_env.observation_space
    
    @property
    def action_space(self):
        return self.hrm_env.action_space