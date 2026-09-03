from pydantic_settings import BaseSettings
from pathlib import Path
import yaml


class Settings(BaseSettings):
    razorpay_key_id: str = "rzp_test_dummy"
    razorpay_key_secret: str = "dummy_secret"
    anthropic_api_key: str = "sk-ant-dummy"

    db_path: str = "recovery_copilot.db"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class PolicyConfig:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "policy.yaml"
        with open(config_path) as f:
            self._data = yaml.safe_load(f)

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    @property
    def max_retries(self) -> int:
        return self._data.get("max_retries_per_transaction", 3)

    @property
    def max_recovery_steps(self) -> int:
        return self._data.get("max_recovery_steps", 3)

    @property
    def min_cooling_hours(self) -> int:
        return self._data.get("min_cooling_hours", 24)

    @property
    def rbi_afa_threshold_paise(self) -> int:
        return self._data.get("rbi_afa_threshold_paise", 1500000)

    @property
    def max_discount_percent(self) -> int:
        return self._data.get("max_discount_percent", 10)

    @property
    def max_contacts_per_week(self) -> int:
        return self._data.get("max_contacts_per_week", 3)

    @property
    def contact_window_start(self) -> str:
        return self._data.get("contact_window_start", "08:00")

    @property
    def contact_window_end(self) -> str:
        return self._data.get("contact_window_end", "21:00")


settings = Settings()
