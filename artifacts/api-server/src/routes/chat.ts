import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { Router, type IRouter } from "express";
import {
  SendChatMessageBody,
  SendChatMessageResponse,
} from "@workspace/api-zod";

const router: IRouter = Router();
const MAX_TOTAL_CHARACTERS = 24_000;
const MAX_OUTPUT_CHARACTERS = 64_000;
const INFERENCE_TIMEOUT_MS = 180_000;

let inferenceActive = false;

class ModelTimeoutError extends Error {
  constructor() {
    super("Local Qwen inference timed out.");
    this.name = "ModelTimeoutError";
  }
}

class ModelProcessError extends Error {
  readonly exitCode: number | null;
  readonly stderr: string;

  constructor(exitCode: number | null, stderr: string) {
    super(`Local Qwen inference exited with code ${exitCode}.`);
    this.name = "ModelProcessError";
    this.exitCode = exitCode;
    this.stderr = stderr;
  }
}

function findWorkspaceRoot(): string {
  let directory = resolve(process.cwd());

  for (let depth = 0; depth < 6; depth += 1) {
    if (existsSync(resolve(directory, "scripts/ask_model.py"))) {
      return directory;
    }
    const parent = dirname(directory);
    if (parent === directory) {
      break;
    }
    directory = parent;
  }

  throw new Error("Could not locate scripts/ask_model.py from the API server.");
}

const workspaceRoot = findWorkspaceRoot();

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

function runLocalInference(messages: ChatMessage[]): Promise<string> {
  return new Promise((resolvePromise, rejectPromise) => {
    const child = spawn(
      "uv",
      [
        "run",
        "python",
        "scripts/ask_model.py",
        "--messages-stdin",
        "--max-new-tokens",
        "160",
      ],
      {
        cwd: workspaceRoot,
        env: { ...process.env, PYTHONUNBUFFERED: "1" },
        stdio: ["pipe", "pipe", "pipe"],
      },
    );

    let stdout = "";
    let stderr = "";
    let timedOut = false;
    let settled = false;
    let forceKillTimer: ReturnType<typeof setTimeout> | undefined;

    const timeoutTimer = setTimeout(() => {
      timedOut = true;
      child.kill("SIGTERM");
      forceKillTimer = setTimeout(() => child.kill("SIGKILL"), 5_000);
    }, INFERENCE_TIMEOUT_MS);

    const settle = (error?: Error, answer?: string) => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timeoutTimer);
      if (forceKillTimer) {
        clearTimeout(forceKillTimer);
      }
      if (error) {
        rejectPromise(error);
      } else {
        resolvePromise(answer ?? "");
      }
    };

    child.stdout.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => {
      stdout += chunk;
      if (stdout.length > MAX_OUTPUT_CHARACTERS) {
        child.kill("SIGKILL");
        settle(new Error("The model output exceeded the response size limit."));
      }
    });

    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk: string) => {
      stderr = `${stderr}${chunk}`.slice(-8_000);
    });

    child.on("error", (error) => settle(error));
    child.on("close", (code) => {
      if (timedOut) {
        settle(new ModelTimeoutError());
        return;
      }
      if (code !== 0) {
        settle(new ModelProcessError(code, stderr));
        return;
      }
      const answer = stdout.trim();
      if (!answer) {
        settle(new Error("The model returned an empty response."));
        return;
      }
      settle(undefined, answer);
    });

    child.stdin.on("error", () => {
      // The close event reports the process exit and any captured diagnostics.
    });
    child.stdin.end(JSON.stringify({ messages }));
  });
}

router.post("/chat", async (req, res): Promise<void> => {
  const parsed = SendChatMessageBody.safeParse(req.body);
  if (!parsed.success) {
    req.log.warn({ error: parsed.error.message }, "Rejected invalid chat request");
    res.status(400).json({ error: "Send a valid conversation with non-empty messages." });
    return;
  }

  const { messages } = parsed.data;
  const totalCharacters = messages.reduce(
    (total, message) => total + message.content.length,
    0,
  );
  const alternatingRoles = messages.every(
    (message, index) => message.role === (index % 2 === 0 ? "user" : "assistant"),
  );

  if (
    totalCharacters > MAX_TOTAL_CHARACTERS ||
    messages[messages.length - 1]?.role !== "user" ||
    !alternatingRoles ||
    messages.some((message) => !message.content.trim())
  ) {
    res.status(400).json({
      error: "Messages must alternate between user and assistant and end with a user message.",
    });
    return;
  }

  if (inferenceActive) {
    res.status(429).json({ error: "The local model is busy. Try again shortly." });
    return;
  }

  inferenceActive = true;
  try {
    const answer = await runLocalInference(messages);
    const response = SendChatMessageResponse.parse({
      message: { role: "assistant", content: answer },
    });
    res.json(response);
  } catch (error) {
    if (error instanceof ModelTimeoutError) {
      req.log.warn({ err: error }, "Local Qwen inference timed out");
      res.status(504).json({ error: "The local model took too long to respond. Please try again." });
      return;
    }

    req.log.error({ err: error }, "Local Qwen inference failed");
    res.status(503).json({
      error: "The local model could not complete this request. Check the model setup and try again.",
    });
  } finally {
    inferenceActive = false;
  }
});

export default router;
