import pytest

from memory_r1.providers import NIM, OPENAI, get_provider, make_client, make_embedder, make_llm


class FakeCompletions:
    def create(self, **kwargs):
        self.kwargs = kwargs

        class Msg:
            content = "hello"

        class Choice:
            message = Msg()

        class Response:
            choices = [Choice()]

        return Response()


class FakeClient:
    def __init__(self):
        self.completions = FakeCompletions()

    @property
    def chat(self):
        return self


def test_default_provider_is_nim(monkeypatch):
    monkeypatch.delenv("MEMR1_PROVIDER", raising=False)
    assert get_provider() is NIM
    assert NIM.base_url == "https://integrate.api.nvidia.com/v1"


def test_provider_env_override(monkeypatch):
    monkeypatch.setenv("MEMR1_PROVIDER", "openai")
    assert get_provider() is OPENAI


def test_explicit_name_beats_env(monkeypatch):
    monkeypatch.setenv("MEMR1_PROVIDER", "openai")
    assert get_provider("nim") is NIM


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        get_provider("bedrock")


def test_make_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
        make_client(NIM)


def test_make_llm_uses_provider_chat_model(monkeypatch):
    monkeypatch.delenv("MEMR1_CHAT_MODEL", raising=False)
    client = FakeClient()
    llm = make_llm(NIM, client=client)
    assert llm("hi") == "hello"
    assert client.completions.kwargs["model"] == NIM.chat_model
    assert client.completions.kwargs["temperature"] == 0.0


def test_make_llm_model_env_override(monkeypatch):
    monkeypatch.setenv("MEMR1_CHAT_MODEL", "qwen/qwen2.5-7b-instruct")
    client = FakeClient()
    make_llm(NIM, client=client)("hi")
    assert client.completions.kwargs["model"] == "qwen/qwen2.5-7b-instruct"


def test_make_embedder_uses_provider_model(monkeypatch):
    monkeypatch.delenv("MEMR1_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("MEMR1_RPM", raising=False)
    embedder = make_embedder(NIM, client=object())
    assert embedder._model == NIM.embedding_model
    assert embedder._input_type_param is True
    assert embedder._throttle is not None
    assert embedder._retry is not None

    embedder = make_embedder(OPENAI, client=object())
    assert embedder._input_type_param is False
    assert embedder._throttle is None


def test_rate_limiter_spaces_calls():
    from memory_r1.providers import RateLimiter

    now = {"t": 100.0}
    sleeps = []

    def clock():
        return now["t"]

    def sleep(s):
        sleeps.append(s)
        now["t"] += s

    limiter = RateLimiter(30, clock=clock, sleep=sleep)  # 2s interval
    limiter.wait()
    assert sleeps == []
    limiter.wait()
    assert sleeps == [2.0]
    now["t"] += 5  # long gap: no sleep needed
    limiter.wait()
    assert sleeps == [2.0]


def test_429_retry_then_success():
    import httpx
    from openai import RateLimitError

    from memory_r1.providers import _with_429_retry

    err = RateLimitError(
        "429",
        response=httpx.Response(429, request=httpx.Request("POST", "http://x")),
        body=None,
    )
    calls = {"n": 0}
    sleeps = []

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise err
        return "ok"

    assert _with_429_retry(flaky, sleep=sleeps.append) == "ok"
    assert sleeps == [15, 30]


def test_429_retry_exhausted_raises():
    import httpx
    import pytest as _pytest
    from openai import RateLimitError

    from memory_r1.providers import _with_429_retry

    err = RateLimitError(
        "429",
        response=httpx.Response(429, request=httpx.Request("POST", "http://x")),
        body=None,
    )

    def always_fail():
        raise err

    with _pytest.raises(RateLimitError):
        _with_429_retry(always_fail, attempts=3, sleep=lambda s: None)
