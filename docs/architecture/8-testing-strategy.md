# 8. Testing Strategy

## 8.1 Unit Testing

### 8.1.1 HRM Component Tests

```python
class TestHierarchicalReasoningModel(unittest.TestCase):
    """Unit tests for HRM core components"""
    
    def setUp(self):
        self.config = self.load_test_config()
        self.model = HierarchicalReasoningModel(self.config)
        self.test_data = self.generate_test_data()
    
    def test_hierarchical_convergence(self):
        """Test hierarchical convergence mechanism"""
        observation = self.test_data['observation']
        
        # Test convergence over multiple cycles
        outputs, final_states = self.model(observation)
        
        # Verify output dimensions
        self.assertEqual(outputs['action_type'].shape[-1], 5)
        self.assertEqual(outputs['quantity'].shape[-1], 1)
        
        # Verify hierarchical state progression
        self.assertIsNotNone(final_states[0])  # H-module state
        self.assertIsNotNone(final_states[1])  # L-module state
    
    def test_act_mechanism(self):
        """Test Adaptive Computation Time mechanism"""
        act = AdaptiveComputationTime(self.config)
        
        # Test halting decisions
        q_values = torch.tensor([0.3, 0.7])  # [halt, continue]
        should_halt, reason = act.should_halt(q_values, segment_count=3, market_complexity=0.5)
        
        self.assertIsInstance(should_halt, bool)
        self.assertIsInstance(reason, str)
    
    def test_inference_performance(self):
        """Test inference latency requirements"""
        observation = self.test_data['observation']
        
        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            outputs, _ = self.model(observation)
        inference_time = (time.time() - start_time) * 1000
        
        # Verify performance requirement
        self.assertLess(inference_time, 50, "Inference time exceeds 50ms requirement")
```

## 8.2 Integration Testing

### 8.2.1 End-to-End Trading Pipeline

```python
class TestHRMTradingPipeline(unittest.TestCase):
    """Integration tests for complete HRM trading pipeline"""
    
    def setUp(self):
        self.trading_service = self.create_test_trading_service()
        self.backtest_service = self.create_test_backtest_service()
        self.mock_market_data = self.load_mock_market_data()
    
    def test_live_trading_integration(self):
        """Test HRM integration with live trading service"""
        # Simulate live market data
        market_data = self.mock_market_data['live_data']
        
        # Process with HRM agent
        action_type, quantity = self.trading_service.agent.select_action(market_data)
        
        # Verify valid trading actions
        self.assertIn(action_type, range(5))  # 0-4 action types
        self.assertGreater(quantity, 0)
        self.assertLessEqual(quantity, 100000)
    
    def test_backtest_compatibility(self):
        """Test HRM compatibility with existing backtest framework"""
        backtest_config = {
            'symbol': 'NIFTY_5',
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'initial_capital': 100000
        }
        
        # Run backtest with HRM
        results = self.backtest_service.run_backtest(backtest_config)
        
        # Verify enhanced results structure
        self.assertIn('total_return', results)
        self.assertIn('hrm_metrics', results)
        self.assertIn('avg_reasoning_depth', results['hrm_metrics'])
```

## 8.3 Performance Testing

### 8.3.1 Load Testing

```python
class TestHRMPerformance(unittest.TestCase):
    """Performance tests for HRM under load"""
    
    def test_concurrent_inference(self):
        """Test HRM performance under concurrent load"""
        num_concurrent_requests = 100
        
        async def inference_task():
            observation = self.generate_random_observation()
            return self.hrm_agent.select_action(observation)
        
        # Run concurrent inference
        start_time = time.time()
        tasks = [inference_task() for _ in range(num_concurrent_requests)]
        results = asyncio.run(asyncio.gather(*tasks))
        total_time = time.time() - start_time
        
        # Verify performance
        avg_time_per_request = (total_time / num_concurrent_requests) * 1000
        self.assertLess(avg_time_per_request, 50, "Average inference time too high")
        
        # Verify all requests succeeded
        self.assertEqual(len(results), num_concurrent_requests)
```

---
