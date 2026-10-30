"""Application Configuration Module using Pydantic Settings v2."""

import json
from typing import List, Literal, Optional, Union
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Telegram Credentials
    bot_token: SecretStr = Field(
        ...,
        description="Telegram Bot API Token from @BotFather",
        alias="BOT_TOKEN",
    )
    admin_user_ids: Union[List[int], str, int] = Field(
        default_factory=list,
        description="List of authorized numeric admin Telegram user IDs (single ID, comma-separated, or JSON list)",
        alias="ADMIN_USER_IDS",
    )
    admin_username: str = Field(
        default="rahmonjon_qosimov",
        description="Telegram username for driver inquiries (without @)",
        alias="ADMIN_USERNAME",
    )

    # Group IDs
    public_group_id: Union[List[Union[int, str]], str, int] = Field(
        default_factory=list,
        description="Public group ID(s), usernames (@group), or invite links to monitor passenger messages",
        alias="PUBLIC_GROUP_ID",
    )
    private_group_id: int = Field(
        default=0,
        description="Private VIP group ID where requests are forwarded",
        alias="PRIVATE_GROUP_ID",
    )

    # Cooldown & deduplication
    user_cooldown_hours: int = Field(
        default=24,
        ge=1,
        description="Hours to skip repeated passenger requests from the same user (1 day / 24h default)",
        alias="USER_COOLDOWN_HOURS",
    )

    # Forwarding behavior
    forward_mode: Literal["copy", "forward"] = Field(
        default="copy",
        description="Forwarding method: 'copy' with deep link or raw 'forward'",
        alias="FORWARD_MODE",
    )

    # Database
    database_url: str = Field(
        default="sqlite+aiosqlite:///./taxi_bot.db",
        description="Async database connection string",
        alias="DATABASE_URL",
    )

    # Timezone & Business Logic
    timezone: str = Field(
        default="Asia/Tashkent",
        description="Standard business timezone",
        alias="TIMEZONE",
    )
    default_subscription_days: int = Field(
        default=30,
        ge=1,
        description="Default subscription duration in days",
        alias="DEFAULT_SUBSCRIPTION_DAYS",
    )

    # Classification & AI
    classifier_type: Literal["deepseek", "claude", "anthropic", "ml", "rule", "hybrid", "llm", "openai"] = Field(
        default="deepseek",
        description="Classifier engine: 'deepseek' (DeepSeek-V3), 'claude' (Anthropic Claude), 'ml' (local ML), 'llm' (OpenAI), or 'rule'",
        alias="CLASSIFIER_TYPE",
    )
    deepseek_api_key: Union[SecretStr, None] = Field(
        default=None,
        description="Optional DeepSeek API key for classification",
        alias="DEEPSEEK_API_KEY",
    )
    deepseek_model: str = Field(
        default="deepseek-chat",
        description="DeepSeek model name (e.g. deepseek-chat)",
        alias="DEEPSEEK_MODEL",
    )
    ml_model_path: str = Field(
        default="app/ml/models/taxi_classifier.joblib",
        description="Path to serialized scikit-learn pipeline",
        alias="ML_MODEL_PATH",
    )
    classifier_confidence_threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Minimum score required to forward a message",
        alias="CLASSIFIER_CONFIDENCE_THRESHOLD",
    )
    enable_llm_classifier: bool = Field(
        default=False,
        description="Whether to use an external LLM for ambiguous requests",
        alias="ENABLE_LLM_CLASSIFIER",
    )
    openai_api_key: Union[SecretStr, None] = Field(
        default=None,
        description="Optional OpenAI API key for LLM classification",
        alias="OPENAI_API_KEY",
    )
    anthropic_api_key: Union[SecretStr, None] = Field(
        default=None,
        description="Optional Anthropic Claude API key",
        alias="ANTHROPIC_API_KEY",
    )
    groq_api_key: Union[SecretStr, None] = Field(
        default=None,
        description="Optional Groq Cloud API key for ultra-fast Speech-to-Text (Whisper)",
        alias="GROQ_API_KEY",
    )
    voice_transcription_enabled: bool = Field(
        default=True,
        description="Whether to transcribe incoming voice messages into text",
        alias="VOICE_TRANSCRIPTION_ENABLED",
    )

    # Telegram Userbot (MTProto)
    userbot_enabled: bool = Field(
        default=False,
        description="Whether to run the userbot listener concurrently",
        alias="USERBOT_ENABLED",
    )
    tg_api_id: Optional[int] = Field(
        default=None,
        description="Telegram API ID from my.telegram.org for userbot",
        alias="TG_API_ID",
    )
    tg_api_hash: Optional[str] = Field(
        default=None,
        description="Telegram API Hash from my.telegram.org for userbot",
        alias="TG_API_HASH",
    )
    userbot_session_name: str = Field(
        default="data/userbot.session",
        description="Path to Telethon session file",
        alias="USERBOT_SESSION_NAME",
    )

    # System Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)",
        alias="LOG_LEVEL",
    )

    @field_validator("admin_user_ids", mode="after")
    @classmethod
    def clean_admin_ids(cls, v: Union[str, int, List[int], List[str]]) -> List[int]:
        """Convert single int, comma-separated strings or JSON arrays into List[int]."""
        if isinstance(v, int):
            return [v]
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return []
            if v.startswith("[") and v.endswith("]"):
                try:
                    parsed = json.loads(v)
                    return [int(x) for x in parsed if str(x).strip()]
                except Exception:
                    pass
            return [int(x.strip()) for x in v.split(",") if x.strip()]
        if isinstance(v, list):
            return [int(x) for x in v]
        return []

    @field_validator("public_group_id", mode="after")
    @classmethod
    def clean_public_ids(cls, v: Union[str, int, List[Union[int, str]]]) -> List[Union[int, str]]:
        """Convert single value, comma-separated strings or JSON arrays into clean target list."""
        if isinstance(v, int):
            return [v] if v != 0 else []
        if isinstance(v, str):
            v = v.strip()
            if not v or v == "0":
                return []
            if v.startswith("[") and v.endswith("]"):
                try:
                    parsed = json.loads(v)
                    cleaned = []
                    for x in parsed:
                        s = str(x).strip()
                        if not s or s == "0":
                            continue
                        try:
                            cleaned.append(int(s))
                        except ValueError:
                            cleaned.append(s)
                    return cleaned
                except Exception:
                    pass
            cleaned = []
            for item in v.split(","):
                s = item.strip()
                if not s or s == "0":
                    continue
                try:
                    cleaned.append(int(s))
                except ValueError:
                    cleaned.append(s)
            return cleaned
        if isinstance(v, list):
            cleaned = []
            for item in v:
                s = str(item).strip()
                if not s or s == "0":
                    continue
                try:
                    cleaned.append(int(s))
                except ValueError:
                    cleaned.append(s)
            return cleaned
        return []

    def get_public_target_strings(self) -> List[str]:
        """Return list of all configured public targets (IDs, usernames, links) as strings."""
        if not self.public_group_id:
            return []
        if isinstance(self.public_group_id, list):
            return [str(x) for x in self.public_group_id]
        return [str(self.public_group_id)]

    def get_public_group_numeric_ids(self) -> List[int]:
        """Return only numeric group IDs from the configuration."""
        if not self.public_group_id:
            return []
        targets = self.public_group_id if isinstance(self.public_group_id, list) else [self.public_group_id]
        res = []
        for t in targets:
            try:
                res.append(int(t))
            except (ValueError, TypeError):
                pass
        return res

    @field_validator("tg_api_id", mode="before")
    @classmethod
    def clean_tg_api_id(cls, v):
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        return int(v)

    @field_validator("tg_api_hash", mode="before")
    @classmethod
    def clean_tg_api_hash(cls, v):
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        return str(v).strip()

    @field_validator("userbot_enabled", mode="before")
    @classmethod
    def clean_userbot_enabled(cls, v):
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes", "on")
        return bool(v)

    def is_monitored_public_group(self, chat_id: int) -> bool:
        """Check if chat_id is in monitored public groups. If empty, monitors all groups."""
        if not self.public_group_id:
            return True
        numeric_ids = self.get_public_group_numeric_ids()
        if not numeric_ids:
            return True
        if chat_id in numeric_ids:
            return True
        s_id = str(chat_id)
        if s_id.startswith("-100"):
            raw_id = int(s_id[4:])
            if raw_id in numeric_ids or -raw_id in numeric_ids:
                return True
        elif not s_id.startswith("-"):
            if int(f"-100{s_id}") in numeric_ids or -int(s_id) in numeric_ids:
                return True
        elif s_id.startswith("-"):
            raw_id = int(s_id[1:])
            if int(f"-100{raw_id}") in numeric_ids or raw_id in numeric_ids:
                return True
        return False

    def is_admin(self, user_id: int) -> bool:
        """Check if a numeric Telegram User ID belongs to an administrator."""
        return user_id in self.admin_user_ids


def get_settings() -> Settings:
    """Factory to load and validate settings from environment or .env file."""
    return Settings()
