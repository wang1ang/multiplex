// Installed by multiplex into ~/.pi/agent/extensions/ — do not edit by hand.
//
// Registers whatever model a local multiplex server is currently serving as the
// "multiplex" provider. pi awaits this factory during startup, so the model is
// visible to /model and --list-models. Servers that are not running are skipped
// silently; run /reload after starting or switching one.
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const PORTS = (process.env.MULTIPLEX_PORTS ?? "8000")
  .split(",")
  .map((entry) => entry.trim())
  .filter(Boolean);

// /v1/models reports only OpenAI's required fields, so everything else is a
// default here rather than something read off the wire.
type ServedModel = { id: string };

const CONTEXT_WINDOW = 32768;

async function fetchServed(port: string) {
  const baseUrl = `http://127.0.0.1:${port}/v1`;
  const response = await fetch(`${baseUrl}/models`, {
    signal: AbortSignal.timeout(2000),
  });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const payload = (await response.json()) as { data?: ServedModel[] };
  return (payload.data ?? []).map((served) => ({ served, baseUrl }));
}

export default async function (pi: ExtensionAPI) {
  const settled = await Promise.allSettled(PORTS.map(fetchServed));
  const found = settled.flatMap((entry) => (entry.status === "fulfilled" ? entry.value : []));
  if (found.length === 0) return;

  pi.registerProvider("multiplex", {
    name: "multiplex (local)",
    baseUrl: found[0]!.baseUrl,
    apiKey: "local",
    api: "openai-completions",
    models: found.map(({ served, baseUrl }) => {
      return {
        id: served.id,
        // Distinct ports serving the same model id would otherwise be
        // indistinguishable in /model.
        name: PORTS.length > 1 ? `${served.id} (${new URL(baseUrl).port})` : served.id,
        baseUrl,
        reasoning: true,
        input: ["text"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: CONTEXT_WINDOW,
        maxTokens: 8192,
        compat: {
          // multiplex maps "developer" to "system" itself, but says so here so
          // pi does not depend on that normalization.
          supportsDeveloperRole: false,
          // Thinking is negotiated through reasoning_effort / enable_thinking on
          // the request body (server.py:_enable_thinking_from_body).
          supportsReasoningEffort: true,
          thinkingFormat: "qwen",
          maxTokensField: "max_tokens",
          // No usage accounting in the SSE stream yet.
          supportsUsageInStreaming: false,
        },
      };
    }),
  });
}
