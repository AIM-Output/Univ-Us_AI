"""설정 — .env 는 이 모듈 안에서만 읽고 os.environ 에 쓰지 않는다 · F9_LLM_* 가 LLM_MAIN_* 보다 먼저."""
import importlib
import os

import pytest

KEYS = ("F9_LLM_BASE_URL", "F9_LLM_API_KEY", "F9_LLM_MODEL", "F9_LLM_TIMEOUT", "F9_LLM",
        "LLM_MAIN_BASE_URL", "LLM_MAIN_API_KEY", "LLM_MAIN_MODEL")


@pytest.fixture
def reload_with(tmp_path, monkeypatch):
    """임시 DATA_DIR(+ 그 부모)의 .env 로 config 를 다시 읽는다. 끝나면 원래 환경으로 되돌린다."""
    from chat import config as C
    data = tmp_path / "data"
    data.mkdir()

    def _reload(data_env: str = "", parent_env: str = "", **env):
        (data / ".env").write_text(data_env, encoding="utf-8")
        (tmp_path / ".env").write_text(parent_env, encoding="utf-8")
        monkeypatch.setenv("F9_DATA_DIR", str(data))
        for k in KEYS:
            monkeypatch.delenv(k, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        return importlib.reload(C)

    yield _reload
    monkeypatch.undo()
    importlib.reload(C)


def test_env_file_is_not_leaked_and_f9_names_win(reload_with):
    C = reload_with(
        data_env="F9_LLM_BASE_URL=http://pod/v1/\nLLM_MAIN_BASE_URL=http://main/v1\nLLM_MAIN_API_KEY=mainkey\n"
                 "F9_LLM_MODEL=univus-f9\nLLM_MAIN_MODEL=other\nF9_LLM_TIMEOUT=7\n",
        parent_env="F9_LLM_API_KEY=parentkey\nF9_LLM_BASE_URL=http://loser/v1\n")
    for k in KEYS:
        assert k not in os.environ                          # 다른 기능·자식 프로세스에 새지 않는다
    assert C.LLM_BASE_URL == "http://pod/v1"                  # 앞의 파일(DATA_DIR/.env)이 이긴다 · 끝 / 를 뗀다
    assert C.LLM_API_KEY == "parentkey"                       # F9_LLM_* (부모 파일) 가 LLM_MAIN_* (앞 파일) 보다 먼저
    assert C.LLM_MODEL == "univus-f9"
    assert C.LLM_TIMEOUT == 7.0
    assert C.llm_configured()


def test_llm_main_is_fallback(reload_with):
    from chat import config as C0
    if any(k.startswith("F9_LLM_") for k in C0._read_env(C0.ROOT / ".env")):
        pytest.skip("F9_Chat_agent/.env 에 F9_LLM_* 가 있어 대체 슬롯을 시험할 수 없다")
    C = reload_with(data_env="LLM_MAIN_BASE_URL=http://main/v1\nLLM_MAIN_API_KEY=k\nLLM_MAIN_MODEL=m\n")
    assert (C.LLM_BASE_URL, C.LLM_API_KEY, C.LLM_MODEL, C.LLM_TIMEOUT) == ("http://main/v1", "k", "m", 60.0)
    assert "LLM_MAIN_API_KEY" not in os.environ


def test_environment_beats_env_file(reload_with, monkeypatch):
    C = reload_with(data_env="F9_LLM_BASE_URL=http://file/v1\nF9_LLM_API_KEY=filekey\nF9_LLM_MODEL=m\n",
                    F9_LLM_BASE_URL="http://env/v1", LLM_MAIN_API_KEY="mainenv")
    assert C.LLM_BASE_URL == "http://env/v1"
    assert C.LLM_API_KEY == "filekey"                         # F9 이름이 먼저 — 파일에 있어도 LLM_MAIN 환경변수보다 앞선다
    assert C.llm_configured()
    monkeypatch.setenv("F9_LLM", "off")
    assert not C.llm_configured()


def test_off_switch_from_env_file(reload_with):
    C = reload_with(data_env="F9_LLM_BASE_URL=http://pod/v1\nF9_LLM_API_KEY=k\nF9_LLM_MODEL=m\nF9_LLM=off\n")
    assert not C.llm_configured()
    assert "F9_LLM" not in os.environ
