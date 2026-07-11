import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class MomentumThreshold(StrictModel):
    type: Literal["momentum_threshold"]
    lookback_days: int = Field(ge=20, le=252)
    threshold: float = Field(ge=-0.10, le=0.30)


class MeanReversionZScore(StrictModel):
    type: Literal["mean_reversion_zscore"]
    lookback_days: int = Field(ge=10, le=120)
    entry_z: float = Field(ge=0.50, le=3.00)


class ChannelBreakout(StrictModel):
    type: Literal["channel_breakout"]
    lookback_days: int = Field(ge=10, le=252)
    buffer_pct: float = Field(ge=0.00, le=0.05)
    confirmation_days: int = Field(ge=1, le=5)


EntryConfig = Annotated[
    MomentumThreshold | MeanReversionZScore | ChannelBreakout,
    Field(discriminator="type"),
]


class StopLossPct(StrictModel):
    type: Literal["stop_loss_pct"]
    stop_pct: float = Field(ge=0.02, le=0.25)


class TrailingStop(StrictModel):
    type: Literal["trailing_stop"]
    trail_pct: float = Field(ge=0.02, le=0.25)


class TimeExit(StrictModel):
    type: Literal["time_exit"]
    max_holding_days: int = Field(ge=3, le=126)


ExitConfig = Annotated[
    StopLossPct | TrailingStop | TimeExit,
    Field(discriminator="type"),
]


class UniverseFilter(StrictModel):
    rank_by: Literal["relative_strength_252d", "long_eur_carry"] = (
        "relative_strength_252d"
    )
    side: Literal["top", "bottom"]
    k: int = Field(ge=1, le=10)


class EqualWeightSizing(StrictModel):
    method: Literal["equal_weight"]
    max_positions: int = Field(ge=1, le=5)


class StrategyConfig(StrictModel):
    entry: EntryConfig
    exit: ExitConfig
    universe_filter: UniverseFilter
    sizing: EqualWeightSizing

    @property
    def parameter_count(self) -> int:
        entry_count = 3 if isinstance(self.entry, ChannelBreakout) else 2
        return entry_count + 1 + 1 + 1

    @property
    def normalized_complexity(self) -> float:
        return self.parameter_count / 8.0

    def canonical_json(self) -> str:
        return self.model_dump_json(exclude_none=True)


class StrategyParser:
    ACTION_CONTRACT = """{
  "entry": {
    "type": "channel_breakout",
    "lookback_days": 252,
    "buffer_pct": 0.05,
    "confirmation_days": 5
  },
  "exit": {
    "type": "trailing_stop",
    "trail_pct": 0.25
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
}"""

    ACTION_RULES = """Use these property and type names verbatim; aliases and alternative layouts are invalid.
Allowed choices and inclusive bounds:
- entry.type exactly "momentum_threshold": lookback_days integer 20 to 252; threshold number -0.10 to 0.30
- entry.type exactly "mean_reversion_zscore": lookback_days integer 10 to 120; entry_z number 0.50 to 3.00
- entry.type exactly "channel_breakout": lookback_days integer 10 to 252; buffer_pct number 0.00 to 0.05; confirmation_days integer 1 to 5
- exit.type exactly "stop_loss_pct": stop_pct number 0.02 to 0.25
- exit.type exactly "trailing_stop": trail_pct number 0.02 to 0.25
- exit.type exactly "time_exit": max_holding_days integer 3 to 126
- universe_filter: rank_by exactly "relative_strength_252d" or "long_eur_carry"; side exactly "top" or "bottom"; k integer 1 to 10. long_eur_carry has no eligible assets on panels without reference rates
- sizing: method exactly "equal_weight"; max_positions integer 1 to 5"""

    @classmethod
    def parse(cls, text: str) -> StrategyConfig:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("response must be a non-empty JSON string")
        stripped = text.strip()
        if stripped.startswith("```") or stripped.endswith("```"):
            raise ValueError("markdown fences are not allowed")
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise ValueError("strategy config must be a JSON object")
        try:
            return StrategyConfig.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(f"strategy schema violation: {exc}") from exc

    @staticmethod
    def completion_text(completion: object) -> str:
        if isinstance(completion, str):
            return completion
        if not isinstance(completion, list):
            return ""
        for message in reversed(completion):
            if isinstance(message, dict):
                content = message.get("content")
            else:
                content = getattr(message, "content", None)
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        parts.append(part["text"])
                    elif isinstance(getattr(part, "text", None), str):
                        parts.append(part.text)
                if parts:
                    return "".join(parts)
        return ""
