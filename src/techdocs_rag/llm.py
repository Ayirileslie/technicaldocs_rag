import logging
import time

log = logging.getLogger(__name__)


class GroqLLM:
    """Thin Groq client with exponential-backoff retries on transient errors."""

    def __init__(self, api_key: str, model: str, max_retries: int = 3):
        import groq

        if not api_key:
            raise RuntimeError("Missing required settings: GROQ_API_KEY (see .env.example)")
        self._groq = groq
        self._client = groq.Groq(api_key=api_key.strip())
        self.model = model
        self.max_retries = max_retries

    def complete(self, system: str, user: str) -> str:
        retryable = (self._groq.RateLimitError, self._groq.APIConnectionError, self._groq.InternalServerError)
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    temperature=0.1,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                )
                return resp.choices[0].message.content or ""
            except retryable as e:
                if attempt == self.max_retries:
                    raise
                delay = 2 ** attempt
                log.warning("groq transient error (%s); retrying in %ss", type(e).__name__, delay)
                time.sleep(delay)
        raise AssertionError("unreachable")