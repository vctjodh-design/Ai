import { lookup as dnsLookup } from "node:dns/promises";
import { isIP } from "node:net";
import { request as httpsRequest, type RequestOptions } from "node:https";

export type ResearchSource = {
  title: string;
  url: string;
  snippet: string;
  contentType: "page_excerpt" | "search_summary";
};

const SEARCH_URL = "https://html.duckduckgo.com/html/";
const MAX_RESULTS = 5;
const MAX_SNIPPET_LENGTH = 900;
const MAX_PAGE_EXCERPT_LENGTH = 1_400;
const MAX_PAGE_BYTES = 1_000_000;
const MAX_REDIRECTS = 3;
const MAX_PAGE_FETCHES = 3;

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
        ndash: "–",
        mdash: "—",
        hellip: "…",
        rsquo: "’",
        lsquo: "‘",
        rdquo: "”",
        ldquo: "“",
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
    sources.push({ title, url, snippet, contentType: "search_summary" });
    if (sources.length === MAX_RESULTS) break;
  }

  return sources;
}

function isPublicIp(address: string): boolean {
  const version = isIP(address);
  if (version === 4) {
    const [a = 0, b = 0, c = 0] = address.split(".").map(Number);
    return !(
      a === 0 ||
      a === 10 ||
      (a === 100 && b >= 64 && b <= 127) ||
      a === 127 ||
      (a === 169 && b === 254) ||
      (a === 172 && b >= 16 && b <= 31) ||
      (a === 192 &&
        ((b === 0 && (c === 0 || c === 2)) ||
          (b === 88 && c === 99) ||
          b === 168)) ||
      (a === 198 && (b === 18 || b === 19 || (b === 51 && c === 100))) ||
      (a === 203 && b === 0 && c === 113) ||
      a >= 224
    );
  }

  if (version === 6) {
    const normalized = address.toLowerCase();
    if (normalized.includes("%") || normalized.startsWith("::ffff:")) return false;
    const firstGroup = Number.parseInt(normalized.split(":")[0] ?? "", 16);
    return (
      firstGroup >= 0x2000 &&
      firstGroup <= 0x3fff &&
      !(firstGroup === 0x2001 && normalized.startsWith("2001:db8:"))
    );
  }

  return false;
}

async function resolvePublicAddress(hostname: string) {
  const host = hostname.replace(/^\[|\]$/g, "").replace(/\.$/, "").toLowerCase();
  if (
    host === "localhost" ||
    host.endsWith(".localhost") ||
    host.endsWith(".local") ||
    host.endsWith(".internal")
  ) {
    throw new Error("Private hosts are not valid research sources.");
  }

  if (isIP(host)) {
    if (!isPublicIp(host)) throw new Error("Private IP addresses are not valid research sources.");
    return { address: host, family: isIP(host) };
  }

  const addresses = await dnsLookup(host, { all: true, verbatim: true });
  if (!addresses.length || addresses.some(({ address }) => !isPublicIp(address))) {
    throw new Error("Research source resolves to a private or reserved address.");
  }
  return addresses[0]!;
}

function extractPageExcerpt(html: string): string {
  const cleaned = html
    .replace(/<!--[\s\S]*?-->/g, " ")
    .replace(/<a\b[^>]*>[\s\S]*?<\/a>/gi, " ")
    .replace(/<(script|style|svg|noscript|iframe|template|nav|header|footer|aside|form)\b[^>]*>[\s\S]*?<\/\1>/gi, " ");
  const mainContent =
    cleaned.match(/<(article|main)\b[^>]*>[\s\S]*?<\/\1>/i)?.[0] ?? cleaned;
  const paragraphs = [
    ...mainContent.matchAll(/<p\b[^>]*>([\s\S]*?)<\/p>/gi),
  ]
    .map((match) => htmlToText(match[1] ?? ""))
    .filter((text) => text.length >= 45 && text.length <= 700);
  const excerpt = paragraphs
    .slice(0, 14)
    .join(" ")
    .slice(0, MAX_PAGE_EXCERPT_LENGTH);
  return excerpt.trim();
}

function extractPageTitle(html: string): string {
  const title = html.match(/<title\b[^>]*>([\s\S]*?)<\/title>/i)?.[1] ?? "";
  return htmlToText(title);
}

type PageResponse = {
  text: string;
  title: string;
  redirect?: string;
};

async function requestPage(
  url: URL,
): Promise<PageResponse> {
  const pinnedAddress = await resolvePublicAddress(url.hostname);

  return new Promise((resolvePromise, rejectPromise) => {
    let settled = false;
    let bytesRead = 0;
    const chunks: Buffer[] = [];
    const settle = (error?: Error, result?: PageResponse) => {
      if (settled) return;
      settled = true;
      if (error) rejectPromise(error);
      else resolvePromise(result ?? { text: "", title: "" });
    };
    const options: RequestOptions = {
      protocol: "https:",
      hostname: url.hostname,
      port: 443,
      method: "GET",
      path: `${url.pathname}${url.search}`,
      headers: {
        accept: "text/html,application/xhtml+xml,text/plain",
        "accept-encoding": "identity",
        "user-agent": "Mozilla/5.0 (compatible; QwenChatResearch/1.0)",
      },
      lookup: (_hostname, lookupOptions, callback) => {
        if (lookupOptions && typeof lookupOptions === "object" && lookupOptions.all) {
          callback(null, [pinnedAddress]);
        } else {
          callback(null, pinnedAddress.address, pinnedAddress.family);
        }
      },
    };

    const request = httpsRequest(options, (response) => {
      const status = response.statusCode ?? 0;
      const location = response.headers.location;
      if (status >= 300 && status < 400 && location) {
        response.resume();
        settle(undefined, { text: "", title: "", redirect: new URL(location, url).href });
        return;
      }
      if (status !== 200) {
        response.resume();
        settle(new Error(`Source page returned status ${status}.`));
        return;
      }

      const contentType = response.headers["content-type"] ?? "";
      if (!/^(text\/html|application\/xhtml\+xml|text\/plain)(?:;|$)/i.test(contentType)) {
        response.resume();
        settle(new Error("Source page is not readable text."));
        return;
      }

      response.on("data", (chunk: Buffer | string) => {
        const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
        bytesRead += buffer.byteLength;
        if (bytesRead > MAX_PAGE_BYTES) {
          response.destroy();
          settle(new Error("Source page exceeded the size limit."));
          return;
        }
        chunks.push(buffer);
      });
      response.on("end", () => {
        const html = Buffer.concat(chunks).toString("utf8");
        settle(undefined, {
          text: extractPageExcerpt(html),
          title: extractPageTitle(html),
        });
      });
      response.on("error", (error) => settle(error));
      response.on("aborted", () => settle(new Error("Source page response was interrupted.")));
    });

    request.setTimeout(7_000, () => request.destroy(new Error("Source page timed out.")));
    request.on("error", (error) => settle(error));
    request.end();
  });
}

async function readPage(rawUrl: string): Promise<{ url: string; title: string; excerpt: string }> {
  let url = new URL(rawUrl);
  for (let redirects = 0; redirects <= MAX_REDIRECTS; redirects += 1) {
    if (
      url.protocol !== "https:" ||
      (url.port && url.port !== "443") ||
      url.username ||
      url.password
    ) {
      throw new Error("Only public HTTPS source pages are supported.");
    }
    const result = await requestPage(url);
    if (!result.redirect) {
      if (result.text.length < 80) {
        throw new Error("Source page did not contain enough readable text.");
      }
      return {
        url: url.href,
        title: result.title || url.hostname,
        excerpt: result.text,
      };
    }
    if (redirects === MAX_REDIRECTS) throw new Error("Source page redirected too many times.");
    url = new URL(result.redirect);
  }
  throw new Error("Source page could not be read.");
}

export async function researchWeb(query: string): Promise<ResearchSource[]> {
  const trimmedQuery = query.trim().slice(0, 1_000);
  if (!trimmedQuery) {
    throw new WebResearchUnavailableError("A non-empty search question is required.");
  }

  const directLinks = [
    ...trimmedQuery.matchAll(/\bhttps?:\/\/[^\s<>"']+/gi),
  ]
    .map(([link]) => link.replace(/[),.;!?]+$/g, ""))
    .filter((link, index, all) => all.indexOf(link) === index)
    .slice(0, MAX_PAGE_FETCHES);

  if (directLinks.length) {
    try {
      return await Promise.all(
        directLinks.map(async (link) => {
          const page = await readPage(link);
          return {
            title: page.title,
            url: page.url,
            snippet: page.excerpt,
            contentType: "page_excerpt" as const,
          };
        }),
      );
    } catch (error) {
      throw new WebResearchUnavailableError(
        error instanceof Error
          ? `Could not read a supplied link: ${error.message}`
          : "Could not read a supplied link.",
      );
    }
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

  return Promise.all(
    sources.map(async (source, index) => {
      if (index >= MAX_PAGE_FETCHES) return source;
      try {
        const page = await readPage(source.url);
        return {
          ...source,
          title: page.title || source.title,
          url: page.url,
          snippet: page.excerpt,
          contentType: "page_excerpt" as const,
        };
      } catch {
        return source;
      }
    }),
  );
}
