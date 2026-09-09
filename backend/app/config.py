"""Central application configuration loading.

``config.yaml`` owns non-secret runtime settings and feature switches.  The
selected profile points to a dotenv file that contains secrets and other
secret-bearing values such as database connection strings.  All application
modules receive the final values through ``os.environ`` for compatibility with
the existing codebase.
"""

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml
from dotenv import load_dotenv


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_FILE = REPOSITORY_ROOT / "config.yaml"

_loaded = False
_config: Dict[str, Any] = {}
_selected_config_path: Optional[Path] = None
_selected_env_path: Optional[Path] = None
_selected_profile: Optional[str] = None


_SETTING_ENV_NAMES = {
    "llm_model": "LLM_MODEL",
    "llm_reasoning_effort": "LLM_REASONING_EFFORT",
    "llm_thinking_type": "LLM_THINKING_TYPE",
    "llm_clear_thinking": "LLM_CLEAR_THINKING",
    "openai_api_base": "OPENAI_API_BASE",
    "embedding_model": "EMBEDDING_MODEL",
    "knowledge_base_dir": "KNOWLEDGE_BASE_DIR",
    "chroma_dir": "CHROMA_DIR",
    "cors_origins": "CORS_ORIGINS",
    "backend_url": "BACKEND_URL",
    "api_base_url": "API_BASE_URL",
    "ssl_verify": "SSL_VERIFY",
    "jwt_expire_hours": "JWT_EXPIRE_HOURS",
    "allow_local_auth": "ALLOW_LOCAL_AUTH",
    "db_pool_size": "DB_POOL_SIZE",
    "db_max_overflow": "DB_MAX_OVERFLOW",
    "db_pool_timeout": "DB_POOL_TIMEOUT",
}

_FEATURE_ENV_NAMES = {
    "rag_startup_index": "RAG_STARTUP_INDEX",
    "reranker": "ENABLE_RERANKER",
    "query_expansion": "ENABLE_QUERY_EXPANSION",
    "hyde": "ENABLE_HYDE",
    "cot": "ENABLE_COT",
    "self_rag": "ENABLE_SELF_RAG",
}


def _resolve_path(value: str, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(
            f"{label} 不存在: {path}。"
            "请检查配置文件中的路径，或创建对应文件。"
        )
    return path


def resolve_config_file() -> Optional[Path]:
    """Resolve the YAML configuration file for this process."""

    configured = os.getenv("FITNESS_CONFIG_FILE")
    if configured:
        return _resolve_path(configured, "配置文件")
    return DEFAULT_CONFIG_FILE if DEFAULT_CONFIG_FILE.is_file() else None


def _read_yaml(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    try:
        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
    except yaml.YAMLError as exc:
        raise RuntimeError(f"配置文件 YAML 格式错误: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"配置文件顶层必须是对象: {path}")
    return data


def _as_env_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def _profile_config(data: Mapping[str, Any]) -> tuple[str, Mapping[str, Any], Mapping[str, Any], Optional[str]]:
    """Return profile name, settings, features, and the configured dotenv path."""

    profiles = data.get("profiles")
    if isinstance(profiles, dict) and profiles:
        environment = data.get("environment") or {}
        if not isinstance(environment, dict):
            raise RuntimeError("config.yaml 的 environment 必须是对象")

        profile_name = (
            os.getenv("FITNESS_PROFILE")
            or data.get("active_profile")
            or environment.get("active")
            or "prod"
        )
        profile = profiles.get(profile_name)
        if not isinstance(profile, dict):
            available = ", ".join(str(name) for name in profiles)
            raise RuntimeError(
                f"未找到配置 profile '{profile_name}'，可用 profile: {available}"
            )

        settings = profile.get("settings") or {}
        features = profile.get("features") or data.get("features") or {}
        if not isinstance(settings, dict) or not isinstance(features, dict):
            raise RuntimeError("config.yaml 中 profile 的 settings/features 必须是对象")
        return str(profile_name), settings, features, profile.get("env_file")

    # Backward-compatible flat format for a small config file.
    environment = data.get("environment") or {}
    if not isinstance(environment, dict):
        raise RuntimeError("config.yaml 的 environment 必须是对象")
    profile_name = str(os.getenv("FITNESS_PROFILE") or environment.get("active") or "default")
    settings = data.get("runtime") or data.get("settings") or {}
    features = data.get("features") or {}
    return profile_name, settings, features, environment.get("env_file")


def resolve_env_file(config: Optional[Mapping[str, Any]] = None) -> Optional[Path]:
    """Resolve the selected dotenv file relative to the repository root.

    ``ENV_FILE`` remains a deliberate emergency/CI override.  Normal local
    and production selection should be made through ``config.yaml`` profiles.
    """

    configured = os.getenv("ENV_FILE")
    if configured:
        return _resolve_path(configured, "环境变量 ENV_FILE 指定的文件")

    if config is not None:
        _, _, _, env_file = _profile_config(config)
        if env_file:
            return _resolve_path(str(env_file), "profile 指定的环境文件")

    default_path = REPOSITORY_ROOT / ".env"
    return default_path if default_path.is_file() else None


def _apply_yaml_settings(settings: Mapping[str, Any], features: Mapping[str, Any]) -> None:
    """Expose YAML settings as environment variables without overriding process env."""

    values: Dict[str, Any] = {}
    for config_name, env_name in _SETTING_ENV_NAMES.items():
        if config_name in settings and settings[config_name] is not None:
            values[env_name] = settings[config_name]
    for config_name, env_name in _FEATURE_ENV_NAMES.items():
        if config_name in features and features[config_name] is not None:
            values[env_name] = features[config_name]

    # Set YAML values before dotenv loading. This makes YAML the source of
    # truth for non-secret settings while preserving explicit process overrides.
    for env_name, value in values.items():
        os.environ.setdefault(env_name, _as_env_value(value))


def load_environment() -> Optional[Path]:
    """Load YAML settings and the selected dotenv file once."""

    global _loaded, _config, _selected_config_path, _selected_env_path, _selected_profile
    if _loaded:
        return _selected_env_path

    _selected_config_path = resolve_config_file()
    _config = _read_yaml(_selected_config_path)
    (
        _selected_profile,
        settings,
        features,
        _profile_env_file,
    ) = _profile_config(_config)
    _apply_yaml_settings(settings, features)

    _selected_env_path = resolve_env_file(_config)
    if _selected_env_path:
        load_dotenv(dotenv_path=_selected_env_path, override=False)
    _loaded = True
    return _selected_env_path


def selected_env_file() -> Optional[Path]:
    """Return the selected dotenv file, if any."""

    if not _loaded:
        load_environment()
    return _selected_env_path


def selected_profile() -> Optional[str]:
    """Return the active configuration profile."""

    if not _loaded:
        load_environment()
    return _selected_profile


def selected_config_file() -> Optional[Path]:
    """Return the selected YAML configuration file, if any."""

    if not _loaded:
        load_environment()
    return _selected_config_path
