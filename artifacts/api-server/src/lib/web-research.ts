export type ResearchSource = {
  title: string;
  url: string;
  snippet: string;
};

const SEARCH_URL = "https://html.duckduckgo.com/html/";
const MAX_RESULTS = 5;
const MAX_SNIPPET_LENGTH = 900;

export class WebResearchUnavailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "WebResearchUnavailableError";
  }
}

function decodeHtmlEntities(value: string): string {
  return value.replace(
    /&(#x[\da-f]+|#\d+|amp|quot|apos|lt|gt|nbsp);/gi,
    (entity, code: string) => {
      if (code[0] === "#") {
        const numeric = code[1]?.toLowerCase() === "x"
          ? Number.parseInt(code.slice(2), 16)
          : Number.parseInt(code.slice(1), 10);
        return Number.isFinite(numeric) ? String.fromCodePoint(numeric) : entity;
      }

      const named: Record<string, string> = {
        amp: "&",
        quot: '"',
        apos: "'",
        lt: "<",
        gt: ">",
        nbsp: " ",
      };
      return named[code.toLowerCase()] ?? entity;
    },
  );
}

function htmlToText(value: string): string {
  return decodeHtmlEntities(value.replace(/<[^>]*>/g, " "))
    .replace(/\s+/g, " ")
    .trim();
}

function resolveResultUrl(rawHref: string): string | null {
  try {
    const href = decodeHtmlEntities(rawHref);
    const parsed = new URL(href, SEARCH_URL);
    const target = parsed.hostname.endsWith("duckduckgo.com")
      ? parsed.searchParams.get("uddg")
      : parsed.href;
    if (!target) return null;

    const result = new URL(target);
    return result.protocol === "https:" ? result.href : null;
  } catch {
    return null;
  }
}

function parseResults(html: string): ResearchSource[] {
  const resultLinks = [
    ...html.matchAll(
      /<a\b(?=[^>]*class=["'][^"']*\bresult__a\b[^"']*["'])[^>]*href=["']([^"']+)["'][^>]*>([\s\S]*?)<\/a>/gi,
    ),
  ];
  const resultSnippets = [
    ...html.matchAll(
      /<a\b(?=[^>]*class=["'][^"']*\bresult__snippet\b[^"']*["'])[^>]*>([\s\S]*?)<\/a>/gi,
    ),
  ];
  const sources: ResearchSource[] = [];
  const seen = new Set<string>();

  for (const [index, link] of resultLinks.entries()) {
    const url = resolveResultUrl(link[1] ?? "");
    const title = htmlToText(link[2] ?? "");
    const snippet = htmlToText(resultSnippets[index]?.[1] ?? "")
      .slice(0, MAX_SNIPPET_LENGTH);

    if (!url || !title || !snippet || seen.has(url)) continue;
    seen.add(url);
    sources.push({ title, url, snippet });
    if (sources.length === MAX_RESULTS) break;
  }

  return sources;
}

export async function searchWeb(query: string): Promise<ResearchSource[]> {
  const trimmedQuery = query.trim().slice(0, 1_000);
  if (!trimmedQuery) {
    throw new WebResearchUnavailableError("A non-empty search question is required.");
  }

  const url = new URL(SEARCH_URL);
  url.searchParams.set("q", trimmedQuery);
  let response: Response;
  try {
    response = await fetch(url, {
      headers: {
        accept: "text/html",
        "user-agent": "Mozilla/5.0 (compatible; QwenChatResearch/1.0)",
      },
      signal: AbortSignal.timeout(12_000),
    });
  } catch (error) {
    throw new WebResearchUnavailableError(
      error instanceof Error ? error.message : "Web search could not be reached.",
    );
  }

  if (!response.ok) {
    throw new WebResearchUnavailableError(`Web search returned status ${response.status}.`);
  }

  let html: string;
  try {
    html = await response.text();
  } catch {
    throw new WebResearchUnavailableError("Web search returned an unreadable response.");
  }
  const sources = parseResults(html);
  if (!sources.length) {
    throw new WebResearchUnavailableError("Web search did not return usable source summaries.");
  }
  return sources;
}
