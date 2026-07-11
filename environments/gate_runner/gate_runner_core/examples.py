"""Small valid strategies used by the zero-auth CLI demonstration."""

DEMO_STRATEGIES = (
    (
        "120-day momentum",
        """{
  "entry": {
    "type": "momentum_threshold",
    "lookback_days": 120,
    "threshold": 0.02
  },
  "exit": {
    "type": "time_exit",
    "max_holding_days": 63
  },
  "universe_filter": {
    "rank_by": "relative_strength_252d",
    "side": "top",
    "k": 5
  },
  "sizing": {
    "method": "equal_weight",
    "max_positions": 5
  }
}""",
    ),
    (
        "10-day concentrated breakout",
        """{
  "entry": {
    "type": "channel_breakout",
    "lookback_days": 10,
    "buffer_pct": 0.0,
    "confirmation_days": 1
  },
  "exit": {
    "type": "trailing_stop",
    "trail_pct": 0.02
  },
  "universe_filter": {
    "rank_by": "relative_strength_252d",
    "side": "top",
    "k": 1
  },
  "sizing": {
    "method": "equal_weight",
    "max_positions": 1
  }
}""",
    ),
)
