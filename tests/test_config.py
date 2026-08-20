from multi_agent_research_lab.core.config import Settings


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.openai_model
    assert settings.max_iterations >= 1
    assert settings.timeout_seconds >= 5
    assert settings.max_retries >= 0
    assert settings.retry_backoff_seconds >= 0


def test_settings_accept_field_names_for_programmatic_overrides() -> None:
    settings = Settings(max_retries=0, retry_backoff_seconds=0)
    assert settings.max_retries == 0
    assert settings.retry_backoff_seconds == 0
