# Providers

27 backends through three wire protocols. One adapter per protocol, so a new
OpenAI-compatible gateway is a config entry rather than new code.

| protocol | adapter | backends |
|---|---|---|
| OpenAI Chat Completions | `providers/openai_compat.py` | OpenAI, Azure OpenAI, Groq, OpenRouter, DeepSeek, Together, Fireworks, Cerebras, SambaNova, SiliconFlow, Qwen/DashScope, Moonshot, Zhipu GLM, Perplexity, Hugging Face, GitHub Models, Novita, Chutes, xAI, Mistral, Ollama, LM Studio, vLLM, any custom gateway |
| Anthropic Messages | `providers/anthropic.py` | Anthropic Claude |
| Gemini v1beta | `providers/gemini.py` | Google Gemini |
| offline | `providers/mock.py` | deterministic mock for `nexus demo` and the test-suite |

## Choosing a model

```
nexus -m anthropic:claude-sonnet-4-5      # explicit provider:model
nexus -m sonnet                           # alias
nexus -m gpt-4o                           # resolved through the catalogue
nexus --provider groq                     # provider default model
```

Inside a session: `/model`, `/models [provider] [--refresh]`, `/provider <key>`.

## Credentials

Checked in this order: `--model provider:…` config entry → `providers.<key>.api_key` →
`providers.<key>.api_key_env` → the provider's own environment variables → `~/.nexus/auth.json`
(written by `nexus auth login`, mode `0600`).

```bash
nexus auth login anthropic           # secure prompt
nexus auth login groq gsk_xxx        # inline
nexus auth list                      # what is configured and where it came from
nexus auth logout openai
```

## Failover

If a provider returns a retryable failure (rate limit, auth, 5xx, model-not-found), NEXUS moves
to the next spec in the chain and tells you:

```bash
nexus --failover "anthropic:sonnet,groq:llama-3.3-70b-versatile,openai:gpt-4o-mini"
nexus config set failover '["anthropic:sonnet","groq:llama-3.3-70b-versatile"]'
```

Inside a session `/failover` shows the effective chain. Transport-level retries (429 with
`Retry-After`, 5xx, timeouts, DNS/connection resets) happen first, with exponential backoff and
jitter, before failover is considered.

## Custom and local endpoints

Any OpenAI-compatible server works:

```bash
nexus config set providers.custom.base_url "http://10.0.0.7:8000/v1"
nexus config set providers.custom.api_key  "whatever"
nexus -m custom:my-model
```

```json
{
  "providers": {
    "vllm":    { "base_url": "http://localhost:8000/v1" },
    "ollama":  { "base_url": "http://localhost:11434/v1" },
    "azure":   { "base_url": "https://YOUR-RESOURCE.openai.azure.com/openai/deployments/YOUR-DEPLOYMENT",
                 "api_key_env": "AZURE_OPENAI_API_KEY",
                 "extra": { "api_version": "2024-10-21", "chat_path": "/chat/completions?api-version=2024-10-21" } },
    "gateway": { "base_url": "https://my-gateway.internal/v1",
                 "headers": { "X-Tenant": "team-a" },
                 "extra": { "use_max_completion_tokens": true, "extra_body": { "top_k": 40 } } }
  }
}
```

Per-provider `extra` keys the adapters understand:

| key | provider | effect |
|---|---|---|
| `chat_path`, `models_path` | openai-compatible | non-standard endpoint paths |
| `stream_options` | openai-compatible | disable `stream_options.include_usage` (Ollama/LM Studio/vLLM) |
| `use_max_completion_tokens` | openai-compatible | send `max_completion_tokens` instead of `max_tokens` |
| `allow_reasoning_effort` | openai-compatible | send `reasoning_effort` for non-`o*` models |
| `api_version`, `org`, `project` | openai/azure | extra headers |
| `anthropic_version`, `beta_headers`, `thinking_budget` | anthropic | protocol version, betas, extended thinking |
| `api_version`, `thinking_budget`, `include_thoughts` | gemini | API version and thinking config |
| `extra_body` | all | merged verbatim into the request body |

## Registered providers

| key | name | env var(s) | default base URL | default model |
|---|---|---|---|---|
| `9router` | 9Router (local gateway) | `NINEROUTER_API_KEY`, `NINEROUTER_BASE_URL_KEY` | `http://localhost:20128/v1` | `kr/claude-sonnet-4.5` |
| `anthropic` | Anthropic (Claude) | `ANTHROPIC_API_KEY` | `https://api.anthropic.com` | `claude-sonnet-4-5` |
| `cerebras` | Cerebras | `CEREBRAS_API_KEY` | `https://api.cerebras.ai/v1` | `llama-3.3-70b` |
| `chutes` | Chutes | `CHUTES_API_KEY` | `https://api.chutes.ai/v1` | `deepseek-r1` |
| `custom` | Custom endpoint | `CUSTOM_API_KEY` | `http://localhost:8000/v1` | `model` |
| `deepseek` | DeepSeek | `DEEPSEEK_API_KEY` | `https://api.deepseek.com/v1` | `deepseek-chat` |
| `fireworks` | Fireworks | `FIREWORKS_API_KEY` | `https://api.fireworks.ai/inference/v1` | `accounts/fireworks/models/llama-v3p1-70b-instruct` |
| `gemini` | Google Gemini | `GEMINI_API_KEY`, `GOOGLE_API_KEY` | `https://generativelanguage.googleapis.com` | `gemini-2.5-pro` |
| `github` | GitHub Models | `GITHUB_TOKEN` | `https://models.inference.ai.azure.com` | `gpt-4o` |
| `groq` | Groq | `GROQ_API_KEY` | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` |
| `huggingface` | Hugging Face Router | `HF_TOKEN`, `HUGGINGFACE_API_KEY` | `https://router.huggingface.co/v1` | `meta-llama/Llama-3.3-70B-Instruct` |
| `lmstudio` | LM Studio (local) | _(none — local)_ | `http://localhost:1234/v1` | `local-model` |
| `mistral` | Mistral AI | `MISTRAL_API_KEY` | `https://api.mistral.ai/v1` | `mistral-large-latest` |
| `mock` | Mock (offline) | _(none — local)_ | `mock://local` | `mock-1` |
| `moonshot` | Moonshot (Kimi) | `MOONSHOT_API_KEY` | `https://api.moonshot.cn/v1` | `moonshot-v1-32k-vision-preview` |
| `novita` | Novita | `NOVITA_API_KEY` | `https://api.novita.ai/v3/openai` | `meta-llama/llama-3.3-70b-instruct` |
| `ollama` | Ollama (local) | _(none — local)_ | `http://localhost:11434/v1` | `llama3.2` |
| `openai` | OpenAI | `OPENAI_API_KEY` | `https://api.openai.com/v1` | `gpt-4o` |
| `openrouter` | OpenRouter | `OPENROUTER_API_KEY` | `https://openrouter.ai/api/v1` | `anthropic/claude-sonnet-4.5` |
| `perplexity` | Perplexity | `PERPLEXITY_API_KEY` | `https://api.perplexity.ai` | `sonar-pro` |
| `qwen` | Alibaba Qwen (DashScope) | `DASHSCOPE_API_KEY`, `QWEN_API_KEY` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` |
| `sambanova` | SambaNova | `SAMBANOVA_API_KEY` | `https://api.sambanova.ai/v1` | `Meta-Llama-3.3-70B-Instruct` |
| `siliconflow` | SiliconFlow | `SILICONFLOW_API_KEY` | `https://api.siliconflow.cn/v1` | `Qwen/Qwen2.5-72B-Instruct` |
| `together` | Together AI | `TOGETHER_API_KEY` | `https://api.together.xyz/v1` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` |
| `vllm` | vLLM (local) | _(none — local)_ | `http://localhost:8000/v1` | `served-model` |
| `xai` | xAI (Grok) | `XAI_API_KEY`, `X_AI_API_KEY` | `https://api.x.ai/v1` | `grok-4-fast-reasoning` |
| `zhipu` | Zhipu GLM | `ZHIPUAI_API_KEY`, `ZHIPU_API_KEY` | `https://open.bigmodel.cn/api/paas/v4` | `glm-4-plus` |

### Catalogue models

| model | provider | context | max output | tools | images | reasoning | in $/Mtok | out $/Mtok |
|---|---|---|---|---|---|---|---|---|
| `claude-3-5-haiku-latest` | anthropic | 200,000 | 8,192 | ✓ |  |  | 0.8 | 4 |
| `claude-haiku-4-5` | anthropic | 200,000 | 32,000 | ✓ | ✓ | ✓ | 1 | 5 |
| `claude-opus-4-1` | anthropic | 200,000 | 32,000 | ✓ | ✓ | ✓ | 15 | 75 |
| `claude-sonnet-4-5` | anthropic | 200,000 | 64,000 | ✓ | ✓ | ✓ | 3 | 15 |
| `deepseek-chat` | deepseek | 64,000 | 8,192 | ✓ |  |  | 0.27 | 1.1 |
| `deepseek-reasoner` | deepseek | 64,000 | 8,192 | ✓ |  | ✓ | 0.27 | 1.1 |
| `gemini-2.0-flash` | gemini | 1,048,576 | 8,192 | ✓ | ✓ |  | 0.1 | 0.4 |
| `gemini-2.5-flash` | gemini | 1,048,576 | 65,536 | ✓ | ✓ | ✓ | 0.3 | 2.5 |
| `gemini-2.5-pro` | gemini | 1,048,576 | 65,536 | ✓ | ✓ | ✓ | 1.25 | 10 |
| `llama-3.1-8b-instant` | groq | 128,000 | 8,192 | ✓ |  |  | 0.05 | 0.08 |
| `llama-3.3-70b-versatile` | groq | 128,000 | 32,768 | ✓ |  |  | 0.59 | 0.79 |
| `mistral-large-latest` | mistral | 128,000 | 8,192 | ✓ |  |  | 2 | 6 |
| `gpt-4.1` | openai | 1,047,576 | 32,768 | ✓ | ✓ |  | 2 | 8 |
| `gpt-4.1-mini` | openai | 1,047,576 | 32,768 | ✓ | ✓ |  | 0.4 | 1.6 |
| `gpt-4.1-nano` | openai | 1,047,576 | 32,768 | ✓ | ✓ |  | 0.1 | 0.4 |
| `gpt-4o` | openai | 128,000 | 16,384 | ✓ | ✓ |  | 2.5 | 10 |
| `gpt-4o-mini` | openai | 128,000 | 16,384 | ✓ | ✓ |  | 0.15 | 0.6 |
| `o3` | openai | 200,000 | 100,000 | ✓ | ✓ | ✓ | 2 | 8 |
| `o3-mini` | openai | 200,000 | 100,000 | ✓ |  | ✓ | 1.1 | 4.4 |
| `o4-mini` | openai | 200,000 | 100,000 | ✓ | ✓ | ✓ | 1.1 | 4.4 |
| `sonar-pro` | perplexity | 200,000 | 8,192 | ✓ |  |  | — | — |
| `qwen-max` | qwen | 32,768 | 8,192 | ✓ |  |  | — | — |
| `qwen-plus` | qwen | 131,072 | 16,384 | ✓ |  |  | 0.4 | 1.2 |
| `grok-4-fast-reasoning` | xai | 2,000,000 | 30,000 | ✓ | ✓ | ✓ | — | — |
| `glm-4-plus` | zhipu | 128,000 | 4,096 | ✓ |  |  | — | — |

### Aliases

| you type | resolves to |
|---|---|
| `4o` | `gpt-4o` |
| `claude` | `claude-sonnet-4-5` |
| `deepseek` | `deepseek-chat` |
| `flash` | `gemini-2.5-flash` |
| `gemini` | `gemini-2.5-pro` |
| `glm` | `glm-4-plus` |
| `gpt-4.1` | `gpt-4.1` |
| `gpt4o` | `gpt-4o` |
| `grok` | `grok-4-fast-reasoning` |
| `haiku` | `claude-haiku-4-5` |
| `kimi` | `moonshot-v1-32k-vision-preview` |
| `llama` | `llama-3.3-70b-versatile` |
| `o3` | `o3` |
| `o4` | `o4-mini` |
| `opus` | `claude-opus-4-1` |
| `pro` | `gemini-2.5-pro` |
| `r1` | `deepseek-reasoner` |
| `sonar` | `sonar-pro` |
| `sonnet` | `claude-sonnet-4-5` |
| `v3` | `deepseek-chat` |

### About the catalogue numbers

Context windows, output limits and prices above are **best-effort public specifications** and
will drift as providers change them. Two rules keep that honest:

- A model that is *not* in the catalogue reports `0` / `unknown` — NEXUS never invents a context
  window, and budgeting is skipped (and reported) instead of using a guessed limit.
- Costs shown by `/usage` are estimates computed from token counts and these numbers.

`nexus models --provider <key> --refresh` fetches the live model list from the provider's own
API, which is the authoritative source for what you can actually call.

## Search backends

`web_search` needs one of these configured (there is no scraping fallback, because a scraper
that breaks silently is worse than an honest error):

| backend | key / config |
|---|---|
| `brave` | `BRAVE_API_KEY` |
| `tavily` | `TAVILY_API_KEY` |
| `serper` | `SERPER_API_KEY` |
| `exa` | `EXA_API_KEY` |
| `searxng` | `SEARXNG_URL` (self-hosted) |
| `duckduckgo` | no key — instant-answer API, limited results |

```bash
export BRAVE_API_KEY=...
nexus config set search.backend brave
```

`web_fetch` needs no key, and refuses private/loopback/link-local/metadata addresses unless you
pass `--allow-private-network`.
