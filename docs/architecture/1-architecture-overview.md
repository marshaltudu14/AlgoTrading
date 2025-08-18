# 1. Architecture Overview

## 1.1 System Context

The algorithmic trading system follows a service-oriented architecture with clear separation between frontend (Next.js), backend (FastAPI), and trading logic. The HRM integration will replace the PPO agent while preserving all existing interfaces.

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Next.js       │    │   FastAPI       │    │   HRM Trading   │
│   Frontend      │◄──►│   Backend       │◄──►│   Engine        │
│                 │    │                 │    │                 │
│ • Dashboard     │    │ • Routes        │    │ • HRM Agent     │
│ • Live Trading  │    │ • WebSockets    │    │ • ACT Mechanism │
│ • Backtesting   │    │ • Authentication│    │ • Deep Training │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                        ┌─────────────────┐
                        │   Fyers API     │
                        │   Integration   │
                        │                 │
                        │ • Live Data     │
                        │ • Order Mgmt    │
                        │ • Portfolio     │
                        └─────────────────┘
```

## 1.2 HRM Core Architecture

The HRM implementation consists of four interconnected components operating in a hierarchical structure:

```
Input Data → Input Network → Hierarchical Processing → Output Network → Trading Actions
              (f_I)           (H-Module & L-Module)     (f_O)
                                      │
                                      ▼
                               ACT Mechanism
                             (Adaptive Halting)
```

**Dual-Module Hierarchy:**
- **H-Module (Strategic)**: High-level planning, market trend analysis, risk assessment
- **L-Module (Tactical)**: Detailed execution, technical analysis, position management

---
