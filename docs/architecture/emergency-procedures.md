# Emergency Procedures

## High Latency Response (>50ms)
1. Check system resource utilization
2. Review inference cache hit rates
3. Reduce HRM computation depth
4. Activate performance optimization mode
5. If persistent, switch to PPO fallback

## Memory Issues (>4GB)
1. Clear inference cache
2. Reduce batch size
3. Enable gradient checkpointing
4. Restart HRM service
5. Monitor for memory leaks

## Model Convergence Issues
1. Check training loss trends
2. Validate data quality
3. Adjust learning rate
4. Apply gradient clipping
5. Restore from last stable checkpoint
