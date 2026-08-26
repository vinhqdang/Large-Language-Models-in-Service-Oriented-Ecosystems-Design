from src.deliberation.llm_client import GeminiClient, LocalHFClient


class _FakeGenAIResponse:
    def __init__(self, text):
        self.text = text


class _FakeModels:
    def __init__(self):
        self.calls = []

    def generate_content(self, model, contents, config=None):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return _FakeGenAIResponse(f"response to: {contents}")


class _FakeGenAIClient:
    def __init__(self):
        self.models = _FakeModels()


def test_gemini_client_generate_sends_prompt_and_returns_text():
    fake_client = _FakeGenAIClient()
    client = GeminiClient(fake_client, model_name="gemini-3.5-flash-lite")

    result = client.generate("Should we use microservices?")

    assert result == "response to: Should we use microservices?"
    assert fake_client.models.calls[0]["model"] == "gemini-3.5-flash-lite"
    assert fake_client.models.calls[0]["contents"] == "Should we use microservices?"
    assert fake_client.models.calls[0]["config"] is None


def test_gemini_client_generate_passes_system_instruction_when_given():
    fake_client = _FakeGenAIClient()
    client = GeminiClient(fake_client, model_name="gemini-3.5-flash-lite")

    client.generate("prompt text", system="You are a performance advocate.")

    config = fake_client.models.calls[0]["config"]
    assert config.system_instruction == "You are a performance advocate."


def test_local_hf_client_generate_builds_chat_messages_and_returns_generator_output():
    seen_messages = []

    def fake_generator(messages):
        seen_messages.append(messages)
        return "generated text"

    client = LocalHFClient(fake_generator)

    result = client.generate("What should we decide?", system="You are an advocate.")

    assert result == "generated text"
    assert seen_messages == [[
        {"role": "system", "content": "You are an advocate."},
        {"role": "user", "content": "What should we decide?"},
    ]]


def test_local_hf_client_generate_without_system_omits_system_message():
    seen_messages = []
    client = LocalHFClient(lambda messages: seen_messages.append(messages) or "x")

    client.generate("no system prompt here")

    assert seen_messages == [[{"role": "user", "content": "no system prompt here"}]]
