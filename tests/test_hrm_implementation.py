#!/usr/bin/env python3
"""
Comprehensive test script for HRM (Hierarchical Reasoning Model) implementation
Tests all components and integration with the trading system.
"""

import os
import sys
import torch
import numpy as np
import logging
from pathlib import Path

# Add root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.models.hrm import HRMTradingAgent, HRMCarry
from src.models.hrm.high_level_module import MarketRegime
from src.models.hrm_trading_environment import HRMTradingEnvironment, HRMTradingWrapper
from src.models.hrm_trainer import HRMTrainer, HRMLossFunction
from src.utils.data_loader import DataLoader
from src.env.trading_mode import TradingMode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HRMTestSuite:
    """Comprehensive test suite for HRM implementation"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_results = {}
        
        # Create test data directory if it doesn't exist
        os.makedirs("data/test/final", exist_ok=True)
        os.makedirs("checkpoints/hrm", exist_ok=True)
        os.makedirs("models/hrm", exist_ok=True)
    
    def run_all_tests(self):
        """Run all HRM tests"""
        logger.info("🚀 Starting HRM Implementation Test Suite")
        logger.info(f"Using device: {self.device}")
        
        tests = [
            self.test_hrm_agent_initialization,
            self.test_hrm_agent_forward_pass,
            self.test_hierarchical_modules,
            self.test_adaptive_computation_time,
            self.test_market_regime_classification,
            self.test_hrm_environment_integration,
            self.test_deep_supervision,
            self.test_loss_function,
            self.test_training_components,
            self.test_model_saving_loading
        ]
        
        passed = 0
        total = len(tests)
        
        for test_func in tests:
            try:
                logger.info(f"Running {test_func.__name__}")
                result = test_func()
                self.test_results[test_func.__name__] = result
                if result:
                    logger.info(f"✅ {test_func.__name__} PASSED")
                    passed += 1
                else:
                    logger.error(f"❌ {test_func.__name__} FAILED")
            except Exception as e:
                logger.error(f"💥 {test_func.__name__} CRASHED: {e}")
                self.test_results[test_func.__name__] = False
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("HRM TEST SUITE SUMMARY")
        logger.info("="*50)
        logger.info(f"Total Tests: {total}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {total - passed}")
        logger.info(f"Success Rate: {passed/total*100:.1f}%")
        
        if passed == total:
            logger.info("🎉 ALL TESTS PASSED! HRM implementation is ready.")
        else:
            logger.warning(f"⚠️  {total - passed} tests failed. Check implementation.")
        
        return passed == total
    
    def test_hrm_agent_initialization(self) -> bool:
        """Test HRM agent initialization"""
        try:
            agent = HRMTradingAgent(
                feature_dim=20,  # Features per timestep
                h_lookback=100,  # H-module lookback
                l_lookback=15,   # L-module lookback
                hidden_dim=64,
                H_layers=2,
                L_layers=2,
                num_heads=4,
                H_cycles=1,
                L_cycles=3,
                halt_max_steps=4,
                device=str(self.device)
            )
            
            # Check agent components
            assert hasattr(agent, 'high_level_module')
            assert hasattr(agent, 'low_level_module')
            assert hasattr(agent, 'act_module')
            assert hasattr(agent, 'input_processor')
            
            # Check parameter count
            total_params = sum(p.numel() for p in agent.parameters())
            assert total_params > 0
            
            logger.info(f"HRM Agent initialized with {total_params:,} parameters")
            return True
            
        except Exception as e:
            logger.error(f"HRM agent initialization failed: {e}")
            return False
    
    def test_hrm_agent_forward_pass(self) -> bool:
        """Test HRM agent forward pass"""
        try:
            agent = HRMTradingAgent(
                feature_dim=20,  # Features per timestep
                h_lookback=100,  # H-module lookback
                l_lookback=15,   # L-module lookback
                hidden_dim=64,
                H_layers=2,
                L_layers=2,
                num_heads=4,
                H_cycles=1,
                L_cycles=3,
                halt_max_steps=4,
                device=str(self.device)
            )
            
            # Create initial carry state
            batch_size = 2
            carry = agent.create_initial_carry(batch_size)
            
            # Test observation (flattened features for 100*20 = 2000 market features + 6 account features)
            observation = torch.randn(batch_size, 2006).to(self.device)
            
            # Forward pass
            new_carry, outputs = agent.forward(carry, observation, training=True)
            
            # Check outputs
            required_outputs = [
                'regime_probabilities', 'risk_parameters', 'signal_strength',
                'action_logits', 'quantity', 'confidence',
                'halt_logits', 'continue_logits'
            ]
            
            for output_key in required_outputs:
                assert output_key in outputs, f"Missing output: {output_key}"
                assert outputs[output_key].shape[0] == batch_size
            
            # Test trading decision extraction
            trading_decision = agent.extract_trading_decision(outputs)
            assert 'action_type' in trading_decision
            assert 'quantity' in trading_decision
            
            logger.info("Forward pass successful, all outputs generated")
            return True
            
        except Exception as e:
            logger.error(f"Forward pass test failed: {e}")
            return False
    
    def test_hierarchical_modules(self) -> bool:
        """Test H and L modules separately"""
        try:
            from src.models.hrm.high_level_module import HighLevelModule
            from src.models.hrm.low_level_module import LowLevelModule
            
            # Test High-Level Module
            h_module = HighLevelModule(
                feature_dim=64,
                lookback_window=100,
                hidden_dim=64,
                num_layers=2,
                num_heads=4
            ).to(self.device)
            
            # Test input should be flattened for H-module (100*64 features)
            test_input = torch.randn(2, 6400).to(self.device)  # [batch, flattened_features]
            h_hidden, h_outputs = h_module(test_input)
            
            assert 'regime_probabilities' in h_outputs
            assert 'risk_parameters' in h_outputs
            assert 'signal_strength' in h_outputs
            assert h_outputs['regime_probabilities'].shape == (2, len(MarketRegime))
            
            # Test Low-Level Module
            l_module = LowLevelModule(
                feature_dim=64,
                lookback_window=15,
                hidden_dim=64,
                num_layers=2,
                num_heads=4,
                strategic_context_dim=64
            ).to(self.device)
            
            # Test input for L-module (15*64 features)
            l_test_input = torch.randn(2, 960).to(self.device)  # [batch, 15*64]
            # Strategic context from H-module
            strategic_context = torch.randn(2, 64).to(self.device) 
            l_hidden, l_outputs = l_module(l_test_input, strategic_context, torch.randn(2, 64).to(self.device))
            
            assert 'action_logits' in l_outputs
            assert 'quantity' in l_outputs
            assert 'confidence' in l_outputs
            assert l_outputs['action_logits'].shape == (2, 5)  # 5 actions
            
            logger.info("Hierarchical modules working correctly")
            return True
            
        except Exception as e:
            logger.error(f"Hierarchical modules test failed: {e}")
            return False
    
    def test_adaptive_computation_time(self) -> bool:
        """Test ACT mechanism"""
        try:
            from src.models.hrm.act_module import AdaptiveComputationTime
            
            act_module = AdaptiveComputationTime(
                strategic_hidden_dim=64,
                tactical_hidden_dim=64,
                regime_dim=5,
                performance_dim=4
            ).to(self.device)
            
            # Create test inputs
            strategic_hidden = torch.randn(2, 64).to(self.device)
            tactical_hidden = torch.randn(2, 64).to(self.device)
            market_regime = torch.randn(2, 5).to(self.device)
            performance_metrics = torch.randn(2, 4).to(self.device)
            
            # Test ACT decision
            act_outputs = act_module(
                strategic_hidden, tactical_hidden, market_regime, performance_metrics,
                step_count=1, max_steps=5
            )
            
            assert 'halt_logits' in act_outputs
            assert 'continue_logits' in act_outputs
            assert act_outputs['halt_logits'].shape == (2, 1)
            assert act_outputs['continue_logits'].shape == (2, 1)
            
            logger.info("ACT mechanism working correctly")
            return True
            
        except Exception as e:
            logger.error(f"ACT test failed: {e}")
            return False
    
    def test_market_regime_classification(self) -> bool:
        """Test market regime classification"""
        try:
            agent = HRMTradingAgent(
                feature_dim=20,
                h_lookback=100,
                l_lookback=15,
                hidden_dim=64,
                H_layers=2,
                L_layers=2,
                num_heads=4,
                device=str(self.device)
            )
            
            # Create test data
            carry = agent.create_initial_carry(1)
            observation = torch.randn(1, 2006).to(self.device)  # 100*20 + 6
            
            # Forward pass
            new_carry, outputs = agent.forward(carry, observation, training=False)
            
            # Test market regime from high level module
            if 'regime_probabilities' in outputs:
                regime_probs = outputs['regime_probabilities'].cpu()
                regime_idx = torch.argmax(regime_probs, dim=1)
                market_analysis = {
                    'market_regime': MarketRegime(regime_idx[0].item()),
                    'regime_confidence': torch.max(regime_probs[0]).item()
                }
            
            assert 'market_regime' in market_analysis
            assert isinstance(market_analysis['market_regime'], MarketRegime)
            assert 'regime_confidence' in market_analysis
            assert 0 <= market_analysis['regime_confidence'] <= 1
            
            logger.info(f"Market regime detected: {market_analysis['market_regime']} "
                       f"(confidence: {market_analysis['regime_confidence']:.3f})")
            return True
            
        except Exception as e:
            logger.error(f"Market regime classification test failed: {e}")
            return False
    
    def test_hrm_environment_integration(self) -> bool:
        """Test HRM integration with trading environment"""
        try:
            # Create test data
            self._create_test_data()
            
            # Initialize environment
            data_loader = DataLoader(final_data_dir="data/test/final")
            
            env = HRMTradingEnvironment(
                data_loader=data_loader,
                symbol="test_symbol",
                initial_capital=10000.0,
                mode=TradingMode.TRAINING,
                hrm_config_path="config/hrm_config.yaml",
                device=str(self.device)
            )
            
            # Reset environment
            observation = env.reset()
            assert observation is not None
            assert len(observation) > 0
            
            # Test step
            next_obs, reward, done, info = env.step()
            
            assert isinstance(reward, (float, np.floating))
            assert isinstance(done, bool)
            assert isinstance(info, dict)
            
            # Check HRM-specific info
            if 'market_regime' in info:
                assert 'regime_confidence' in info
            
            logger.info("HRM environment integration successful")
            return True
            
        except Exception as e:
            logger.error(f"Environment integration test failed: {e}")
            return False
    
    def test_deep_supervision(self) -> bool:
        """Test deep supervision mechanism"""
        try:
            # This is a simplified test of the deep supervision concept
            env = self._create_test_environment()
            observation = env.reset()
            
            # Test multiple segments
            segment_count = 0
            max_segments = 3
            
            while segment_count < max_segments:
                next_obs, reward, done, info = env.step()
                segment_count += 1
                
                # Check for segment information
                if 'hrm_segments_used' in info:
                    assert info['hrm_segments_used'] >= 1
                
                if done:
                    break
            
            logger.info(f"Deep supervision test completed with {segment_count} segments")
            return True
            
        except Exception as e:
            logger.error(f"Deep supervision test failed: {e}")
            return False
    
    def test_loss_function(self) -> bool:
        """Test HRM loss function"""
        try:
            import yaml
            
            # Load config
            with open("config/hrm_config.yaml", 'r') as f:
                config = yaml.safe_load(f)
            
            loss_function = HRMLossFunction(config)
            
            # Create test outputs and targets
            batch_size = 2
            outputs = {
                'regime_probabilities': torch.randn(batch_size, 5).to(self.device),
                'action_logits': torch.randn(batch_size, 5).to(self.device),
                'quantity': torch.randn(batch_size, 1).to(self.device),
                'halt_logits': torch.randn(batch_size, 1).to(self.device),
                'continue_logits': torch.randn(batch_size, 1).to(self.device)
            }
            
            targets = {
                'true_regime': torch.randint(0, 5, (batch_size,)).to(self.device),
                'true_action': torch.randint(0, 5, (batch_size,)).to(self.device),
                'true_quantity': torch.randn(batch_size, 1).to(self.device)
            }
            
            segment_rewards = [0.1, -0.05, 0.2]
            
            # Calculate loss
            total_loss, loss_components = loss_function.calculate_loss(
                outputs, targets, segment_rewards
            )
            
            assert isinstance(total_loss, torch.Tensor)
            assert total_loss.requires_grad
            assert 'total_loss' in loss_components
            
            logger.info(f"Loss function working, total loss: {total_loss.item():.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Loss function test failed: {e}")
            return False
    
    def test_training_components(self) -> bool:
        """Test training-related components"""
        try:
            # Test trainer initialization
            trainer = HRMTrainer(
                config_path="config/hrm_config.yaml",
                data_path="data/test/final",
                device=str(self.device)
            )
            
            # Create minimal test data
            self._create_test_data()
            
            # Setup training (this should not crash)
            trainer.setup_training("test_symbol")
            
            assert trainer.model is not None
            assert trainer.optimizer is not None
            
            logger.info("Training components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Training components test failed: {e}")
            return False
    
    def test_model_saving_loading(self) -> bool:
        """Test model saving and loading"""
        try:
            # Create and train a simple agent
            agent = HRMTradingAgent(
                feature_dim=20,
                h_lookback=100,
                l_lookback=15,
                hidden_dim=64,
                H_layers=2,
                L_layers=2,
                num_heads=4,
                device=str(self.device)
            )
            
            # Save model
            save_path = "models/hrm/test_model.pt"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            torch.save({
                'model_state_dict': agent.state_dict(),
                'config': {'test': True}
            }, save_path)
            
            # Load model
            checkpoint = torch.load(save_path, map_location=self.device)
            
            new_agent = HRMTradingAgent(
                observation_dim=100,
                hidden_dim=64,
                H_layers=2,
                L_layers=2,
                num_heads=4
            ).to(self.device)
            
            new_agent.load_state_dict(checkpoint['model_state_dict'])
            
            # Test that models produce same output
            test_input = torch.randn(1, 2006).to(self.device)  # 100*20 + 6
            carry = agent.create_initial_carry(1)
            
            with torch.no_grad():
                _, outputs1 = agent.forward(carry, test_input, training=False)
                _, outputs2 = new_agent.forward(carry, test_input, training=False)
            
            # Check that outputs are similar (allowing for small numerical differences)
            for key in outputs1:
                if key in outputs2:
                    diff = torch.abs(outputs1[key] - outputs2[key]).mean().item()
                    assert diff < 1e-5, f"Large difference in {key}: {diff}"
            
            logger.info("Model saving/loading successful")
            return True
            
        except Exception as e:
            logger.error(f"Model saving/loading test failed: {e}")
            return False
    
    def _create_test_data(self):
        """Create minimal test data for testing"""
        
        test_data_path = "data/test/final/features_test_symbol.csv"
        os.makedirs(os.path.dirname(test_data_path), exist_ok=True)
        
        # Create synthetic OHLCV data with technical indicators
        np.random.seed(42)  # For reproducible tests
        
        n_samples = 1000
        data = {
            'datetime_epoch': np.arange(n_samples),
            'open': 100 + np.cumsum(np.random.randn(n_samples) * 0.1),
            'high': None,  # Will be calculated
            'low': None,   # Will be calculated
            'close': None, # Will be calculated
            'volume': np.random.randint(1000, 10000, n_samples),
        }
        
        # Ensure OHLC relationships are valid
        base_price = data['open']
        data['close'] = base_price + np.random.randn(n_samples) * 0.05
        data['high'] = np.maximum(data['open'], data['close']) + np.abs(np.random.randn(n_samples) * 0.02)
        data['low'] = np.minimum(data['open'], data['close']) - np.abs(np.random.randn(n_samples) * 0.02)
        
        # Add technical indicators
        data['sma_20'] = np.convolve(data['close'], np.ones(20)/20, mode='same')
        data['sma_50'] = np.convolve(data['close'], np.ones(50)/50, mode='same')
        data['rsi_14'] = 50 + np.random.randn(n_samples) * 10  # Simplified RSI
        data['macd'] = np.random.randn(n_samples) * 0.1
        data['atr_14'] = np.abs(np.random.randn(n_samples) * 0.5)
        data['volume_sma'] = np.convolve(data['volume'], np.ones(20)/20, mode='same')
        
        # Create DataFrame and save
        import pandas as pd
        df = pd.DataFrame(data)
        df.to_csv(test_data_path, index=False)
        
        logger.info(f"Test data created: {test_data_path}")
    
    def _create_test_environment(self):
        """Create test environment for testing"""
        self._create_test_data()
        
        data_loader = DataLoader(final_data_dir="data/test/final")
        
        return HRMTradingEnvironment(
            data_loader=data_loader,
            symbol="test_symbol",
            initial_capital=10000.0,
            mode=TradingMode.TRAINING,
            hrm_config_path="config/hrm_config.yaml",
            device=str(self.device)
        )


def main():
    """Main test runner"""
    test_suite = HRMTestSuite()
    success = test_suite.run_all_tests()
    
    if success:
        logger.info("\n🎯 HRM Implementation is ready for use!")
        logger.info("Next steps:")
        logger.info("1. Run training with: python src/models/hrm_trainer.py")
        logger.info("2. Monitor performance and adjust hyperparameters")
        logger.info("3. Compare with baseline models")
    else:
        logger.error("\n❌ HRM Implementation has issues that need to be fixed.")
        logger.error("Please review the failed tests and fix the issues.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)