import type { NextFunction, Request, Response } from 'express';
import { randomUUID } from 'node:crypto';

// Self-contained Langfuse tracing for the chat surfaces. Deliberately has no
// imports from the rest of the server and no npm dependency: it captures the
// already-parsed request body and the raw response bytes, reconstructs the
// completion (JSON or SSE, OpenAI or Anthropic shape), and posts a
// trace+generation pair to the Langfuse ingestion API. Keeping it decoupled
// is what makes this fork's diff a two-file rebase against upstream.

const MAX_CAPTURE_BYTES = 1_000_000; // cap per response; past this we keep usage/metadata only
const FLUSH_INTERVAL_MS = 5_000;
const MAX_BATCH = 50;

type LangfuseConfig = { secretKey: string; publicKey: string; baseUrl: string };

function getConfig(): LangfuseConfig | null {
  const secretKey = process.env.LANGFUSE_SECRET_KEY?.trim();
  const publicKey = process.env.LANGFUSE_PUBLIC_KEY?.trim();
  if (!secretKey || !publicKey) return null;
  const baseUrl = (process.env.LANGFUSE_BASE_URL?.trim() || 'https://cloud.langfuse.com')
    .replace(/\/+$/, '');
  return { secretKey, publicKey, baseUrl };
}

// ---------------------------------------------------------------------------
// Event queue. Batched so a burst of requests does not become a burst of
// HTTP calls; failures are logged once per flush and dropped — tracing must
// never affect serving.
// ---------------------------------------------------------------------------

const queue: object[] = [];
let timer: NodeJS.Timeout | null = null;

function enqueue(events: object[], cfg: LangfuseConfig): void {
  queue.push(...events);
  if (queue.length >= MAX_BATCH) {
    void flush(cfg);
    return;
  }
  if (!timer) {
    timer = setTimeout(() => void flush(cfg), FLUSH_INTERVAL_MS);
    timer.unref();
  }
}

async function flush(cfg: LangfuseConfig): Promise<void> {
  if (timer) { clearTimeout(timer); timer = null; }
  if (queue.length === 0) return;
  const batch = queue.splice(0, queue.length);
  try {
    const res = await fetch(`${cfg.baseUrl}/api/public/ingestion`, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        authorization: 'Basic ' + Buffer.from(`${cfg.publicKey}:${cfg.secretKey}`).toString('base64'),
      },
      body: JSON.stringify({ batch }),
      signal: AbortSignal.timeout(10_000),
    });
    if (!res.ok && res.status !== 207) {
      console.error(`[langfuse] ingestion rejected: HTTP ${res.status} ${(await res.text()).slice(0, 200)}`);
    }
  } catch (err: any) {
    console.error(`[langfuse] ingestion failed: ${err?.message ?? err}`);
  }
}

// ---------------------------------------------------------------------------
// Response reconstruction. The proxy answers in whatever wire shape the
// caller spoke: OpenAI JSON, OpenAI SSE, Anthropic JSON, Anthropic SSE.
// ---------------------------------------------------------------------------

type Extracted = {
  output: unknown;
  usage: { input?: number; output?: number } | null;
  model: string | null;
};

function extractFromJson(body: any): Extracted {
  // Anthropic Messages shape: { content: [{type:'text',text}...], usage:{input_tokens,output_tokens} }
  if (Array.isArray(body?.content)) {
    return {
      output: body.content,
      usage: body.usage
        ? { input: body.usage.input_tokens, output: body.usage.output_tokens }
        : null,
      model: body.model ?? null,
    };
  }
  // OpenAI chat shape: { choices: [{message}], usage:{prompt_tokens,completion_tokens} }
  const message = body?.choices?.[0]?.message;
  return {
    output: message ?? body ?? null,
    usage: body?.usage
      ? { input: body.usage.prompt_tokens, output: body.usage.completion_tokens }
      : null,
    model: body?.model ?? null,
  };
}

function extractFromSse(raw: string): Extracted {
  let text = '';
  let model: string | null = null;
  let usage: Extracted['usage'] = null;
  const toolCalls = new Map<number, { name?: string; args: string }>();
  for (const line of raw.split('\n')) {
    if (!line.startsWith('data:')) continue;
    const payload = line.slice(5).trim();
    if (!payload || payload === '[DONE]') continue;
    let frame: any;
    try { frame = JSON.parse(payload); } catch { continue; }
    model ??= frame.model ?? frame.message?.model ?? null;
    // OpenAI delta frames
    const delta = frame.choices?.[0]?.delta;
    if (delta?.content) text += delta.content;
    for (const tc of delta?.tool_calls ?? []) {
      const slot = toolCalls.get(tc.index ?? 0) ?? { args: '' };
      if (tc.function?.name) slot.name = tc.function.name;
      if (tc.function?.arguments) slot.args += tc.function.arguments;
      toolCalls.set(tc.index ?? 0, slot);
    }
    if (frame.usage) {
      usage = { input: frame.usage.prompt_tokens, output: frame.usage.completion_tokens };
    }
    // Anthropic stream frames
    if (frame.type === 'content_block_delta' && frame.delta?.type === 'text_delta') {
      text += frame.delta.text;
    }
    if (frame.type === 'message_start' && frame.message?.usage) {
      usage = { ...usage, input: frame.message.usage.input_tokens };
    }
    if (frame.type === 'message_delta' && frame.usage) {
      usage = { ...usage, output: frame.usage.output_tokens };
    }
  }
  const output: any = { content: text };
  if (toolCalls.size > 0) {
    output.tool_calls = [...toolCalls.entries()].map(([, v]) => ({
      name: v.name,
      arguments: v.args,
    }));
  }
  return { output, usage, model };
}

// ---------------------------------------------------------------------------
// Middleware
// ---------------------------------------------------------------------------

export function langfuseTraceMiddleware(req: Request, res: Response, next: NextFunction): void {
  const cfg = getConfig();
  if (!cfg || req.method !== 'POST') { next(); return; }

  const startedAt = new Date();
  const chunks: Buffer[] = [];
  let captured = 0;
  let overflowed = false;

  const keep = (chunk: unknown): void => {
    if (overflowed || chunk == null) return;
    const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    captured += buf.length;
    if (captured > MAX_CAPTURE_BYTES) { overflowed = true; chunks.length = 0; return; }
    chunks.push(buf);
  };

  const origWrite = res.write.bind(res);
  const origEnd = res.end.bind(res);
  res.write = ((chunk: any, ...args: any[]) => { keep(chunk); return (origWrite as any)(chunk, ...args); }) as any;
  res.end = ((chunk: any, ...args: any[]) => { keep(chunk); return (origEnd as any)(chunk, ...args); }) as any;

  res.on('finish', () => {
    setImmediate(() => {
      try {
        const body = req.body ?? {};
        const raw = Buffer.concat(chunks).toString('utf8');
        const isSse = (res.getHeader('content-type') ?? '').toString().includes('text/event-stream');
        let extracted: Extracted = { output: null, usage: null, model: null };
        if (!overflowed && raw) {
          if (isSse) {
            extracted = extractFromSse(raw);
          } else {
            try { extracted = extractFromJson(JSON.parse(raw)); } catch { /* non-JSON error body */ }
          }
        }
        const routedVia = res.getHeader('x-routed-via')?.toString() ?? null;
        const traceId = randomUUID();
        const now = new Date().toISOString();
        const input = {
          ...(body.system ? { system: body.system } : {}),
          messages: body.messages ?? null,
          ...(body.tools?.length ? { tools: body.tools.map((t: any) => t?.function?.name ?? t?.name ?? 'tool') } : {}),
        };
        const ok = res.statusCode < 400;
        enqueue([
          {
            id: randomUUID(),
            type: 'trace-create',
            timestamp: now,
            body: {
              id: traceId,
              name: `freellmapi ${req.baseUrl}${req.path}`,
              timestamp: startedAt.toISOString(),
              input,
              output: extracted.output,
              tags: ['freellmapi'],
              metadata: {
                requestedModel: body.model ?? null,
                routedVia,
                statusCode: res.statusCode,
                stream: Boolean(body.stream),
                truncated: overflowed || undefined,
              },
            },
          },
          {
            id: randomUUID(),
            type: 'generation-create',
            timestamp: now,
            body: {
              traceId,
              name: 'chat',
              startTime: startedAt.toISOString(),
              endTime: new Date().toISOString(),
              model: extracted.model ?? routedVia ?? body.model ?? 'unknown',
              input,
              output: extracted.output,
              level: ok ? 'DEFAULT' : 'ERROR',
              statusMessage: ok ? undefined : `HTTP ${res.statusCode}`,
              usage: extracted.usage ?? undefined,
            },
          },
        ], cfg);
      } catch (err: any) {
        console.error(`[langfuse] trace capture failed: ${err?.message ?? err}`);
      }
    });
  });

  next();
}

export function isLangfuseEnabled(): boolean {
  return getConfig() != null;
}
