# 5. Monitoring and Observability

## 5.1 Hierarchical Performance Metrics

### 5.1.1 Real-time Monitoring

```python
class HierarchicalMetricsCollector:
    """Collect and monitor HRM-specific performance metrics"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.convergence_tracker = ConvergenceTracker()
        self.efficiency_calculator = EfficiencyCalculator()
    
    def track_inference_metrics(self, outputs, computation_segments, inference_time):
        """Track real-time inference metrics"""
        metrics = {
            'inference_time_ms': inference_time * 1000,
            'computation_segments': computation_segments,
            'hierarchical_efficiency': self.efficiency_calculator.compute_efficiency(
                outputs, computation_segments
            ),
            'convergence_stability': self.convergence_tracker.stability_score,
            'memory_usage_mb': torch.cuda.memory_allocated() / 1e6,
            'timestamp': datetime.now()
        }
        
        # Performance validation
        self.validate_performance_requirements(metrics)
        
        return metrics
    
    def validate_performance_requirements(self, metrics):
        """Validate against NFR requirements"""
        violations = []
        
        if metrics['inference_time_ms'] > 50:  # NFR1.1
            violations.append(f"Inference time {metrics['inference_time_ms']:.2f}ms > 50ms")
        
        if metrics['memory_usage_mb'] > 4000:  # NFR1.3
            violations.append(f"Memory usage {metrics['memory_usage_mb']:.0f}MB > 4GB")
        
        if violations:
            logger.warning(f"Performance violations: {violations}")
            self.trigger_performance_alerts(violations)
```

### 5.1.2 Dashboard Integration

```python
class HRMDashboardMetrics:
    """Integration with existing dashboard for HRM metrics"""
    
    def __init__(self, websocket_manager):
        self.websocket_manager = websocket_manager
        self.metrics_buffer = deque(maxlen=1000)
    
    async def stream_hrm_metrics(self, metrics):
        """Stream HRM metrics to frontend dashboard"""
        dashboard_data = {
            'type': 'hrm_metrics',
            'data': {
                'hierarchical_efficiency': metrics['hierarchical_efficiency'],
                'computation_depth': metrics['computation_segments'],
                'convergence_stability': metrics['convergence_stability'],
                'inference_performance': {
                    'latency_ms': metrics['inference_time_ms'],
                    'memory_mb': metrics['memory_usage_mb'],
                    'status': 'optimal' if metrics['inference_time_ms'] < 50 else 'warning'
                },
                'adaptive_reasoning': {
                    'market_complexity': metrics.get('market_complexity', 0.5),
                    'reasoning_depth': metrics['computation_segments'],
                    'efficiency_score': metrics['hierarchical_efficiency']
                }
            },
            'timestamp': metrics['timestamp'].isoformat()
        }
        
        # Broadcast to connected clients
        await self.websocket_manager.broadcast_to_users(dashboard_data)
```

## 5.2 Error Handling and Recovery

### 5.2.1 Graceful Degradation

```python
class HRMErrorHandler:
    """Handle HRM-specific errors with graceful degradation"""
    
    def __init__(self, fallback_agent=None):
        self.fallback_agent = fallback_agent  # Simple rule-based agent
        self.error_count = defaultdict(int)
        self.max_errors = 5
    
    def handle_inference_error(self, error, observation):
        """Handle inference errors with fallback strategies"""
        error_type = type(error).__name__
        self.error_count[error_type] += 1
        
        logger.error(f"HRM inference error: {error}")
        
        # Fallback strategies
        if self.error_count[error_type] < self.max_errors:
            # Try simplified HRM inference
            try:
                return self.simplified_inference(observation)
            except Exception as fallback_error:
                logger.error(f"Simplified inference failed: {fallback_error}")
        
        # Final fallback to safe action
        if self.fallback_agent:
            return self.fallback_agent.select_action(observation)
        else:
            return 4, 1.0  # HOLD action with minimal quantity
    
    def simplified_inference(self, observation):
        """Simplified HRM inference with reduced computation"""
        # Single-cycle inference without ACT
        simplified_config = self.create_simplified_config()
        return self.hrm_model.simple_forward(observation, simplified_config)
```

---
