import { useCallback, useEffect, useRef, useState, type FormEvent, type KeyboardEvent, type ReactNode } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Activity, ArrowUp, Bot, Check, CircleAlert, ExternalLink, LoaderCircle, Plus, Search, Sparkles, UserRound } from 'lucide-react';
import { getHealthCheckQueryKey, useHealthCheck, useSendChatMessage } from '@workspace/api-client-react';
import { ErrorBoundary } from '@/components/error-boundary';
import { Toaster } from '@/components/ui/toaster';
import { TooltipProvider } from '@/components/ui/tooltip';
import NotFound from '@/pages/not-found';
import { Route, Switch, useLocation, Router as WouterRouter } from 'wouter';

type AnswerMode = 'local' | 'research';
type Source = { title: string; url: string; snippet: string; contentType: 'page_excerpt' | 'search_summary' };
type Message = { role: 'user' | 'assistant'; content: string; sources?: Source[] };
const STORAGE_KEY = 'qwen-chat-conversation-v1';
const MAX_CONTEXT_MESSAGES = 23;
const MAX_CONTEXT_CHARACTERS = 24_000;

function parseSource(value: unknown): Source | null {
  if (!value || typeof value !== 'object') return null;
  const source = value as Record<string, unknown>;
  if (
    typeof source.title !== 'string' ||
    typeof source.url !== 'string' ||
    typeof source.snippet !== 'string'
  ) return null;
  try {
    if (new URL(source.url).protocol !== 'https:') return null;
    return {
      title: source.title,
      url: source.url,
      snippet: source.snippet,
      contentType: source.contentType === 'page_excerpt' ? 'page_excerpt' : 'search_summary',
    };
  } catch {
    return null;
  }
}

function loadConversation(): Message[] {
  try {
    const value: unknown = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '[]');
    if (!Array.isArray(value)) return [];
    return value.flatMap((item: unknown) => {
      if (!item || typeof item !== 'object') return [];
      const message = item as Record<string, unknown>;
      if (
        (message.role !== 'user' && message.role !== 'assistant') ||
        typeof message.content !== 'string'
      ) return [];
      const sources = Array.isArray(message.sources)
        ? message.sources.map(parseSource).filter((source): source is Source => !!source)
        : undefined;
      return [{
        role: message.role,
        content: message.content,
        ...(sources?.length ? { sources } : {}),
      } as Message];
    });
  } catch {
    return [];
  }
}

function buildContext(history: Message[]): Message[] {
  const context: Message[] = [];
  let characters = 0;

  for (let index = history.length - 1; index >= 0; index -= 1) {
    const message = history[index];
    if (
      context.length >= MAX_CONTEXT_MESSAGES ||
      characters + message.content.length > MAX_CONTEXT_CHARACTERS
    ) {
      break;
    }
    context.unshift(message);
    characters += message.content.length;
  }

  if (context[0]?.role === 'assistant') context.shift();
  return context;
}

function ChatHome() {
  const [messages, setMessages] = useState<Message[]>(loadConversation);
  const [mode, setMode] = useState<AnswerMode>('local');
  const [draft, setDraft] = useState('');
  const [requestError, setRequestError] = useState('');
  const [failedRequest, setFailedRequest] = useState<{ history: Message[]; mode: AnswerMode } | null>(null);
  const textAreaRef = useRef<HTMLTextAreaElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const latestMessagesRef = useRef(messages);
  const health = useHealthCheck({
    query: {
      queryKey: getHealthCheckQueryKey(),
      refetchInterval: 30_000,
      retry: false,
    },
  });
  const chatMutation = useSendChatMessage();
  const isSending = chatMutation.isPending;

  useEffect(() => {
    latestMessagesRef.current = messages;
    try {
      if (messages.length) localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
      else localStorage.removeItem(STORAGE_KEY);
    } catch {
      // The live conversation remains usable if browser storage is unavailable.
    }
  }, [messages]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages, isSending, requestError]);

  useEffect(() => {
    const element = textAreaRef.current;
    if (!element) return;
    element.style.height = '0px';
    element.style.height = `${Math.min(element.scrollHeight, 180)}px`;
  }, [draft]);

  const requestReply = useCallback((history: Message[], requestMode: AnswerMode) => {
    setRequestError('');
    setFailedRequest(null);
    const context = buildContext(history);
    chatMutation.mutate(
      { data: { messages: context, mode: requestMode } },
      {
        onSuccess: (output) => {
          setMessages((current) => [...current, output.message]);
        },
        onError: (error: unknown) => {
          const detail = error instanceof Error ? error.message : '';
          setRequestError(detail ? `The chat service couldn't complete that reply. ${detail}` : 'The chat service couldn’t complete that reply. Check the server and try again.');
          setFailedRequest({ history, mode: requestMode });
        },
      },
    );
  }, [chatMutation.mutate]);

  const submitMessage = useCallback((content: string) => {
    const trimmed = content.trim();
    if (!trimmed || isSending) return;
    const nextHistory = [...latestMessagesRef.current, { role: 'user' as const, content: trimmed }];
    latestMessagesRef.current = nextHistory;
    setMessages(nextHistory);
    setDraft('');
    setRequestError('');
    requestReply(nextHistory, mode);
  }, [isSending, mode, requestReply]);

  const onSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    submitMessage(draft);
  };

  const onComposerKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      submitMessage(draft);
    }
  };

  const startNewChat = () => {
    if (isSending) return;
    setMessages([]);
    latestMessagesRef.current = [];
    setDraft('');
    setRequestError('');
    setFailedRequest(null);
    chatMutation.reset();
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {
      // Ignore storage failures; the in-memory conversation is cleared.
    }
    textAreaRef.current?.focus();
  };

  const serverState = health.isLoading
    ? 'checking'
    : health.isSuccess
      ? 'online'
      : 'offline';

  return (
    <main className="app-shell" data-testid="page-chat">
      <header className="topbar">
        <div className="flex items-center gap-3">
          <div className="brand-mark" aria-hidden="true"><Activity size={17} strokeWidth={2.2} /></div>
          <span className="brand-name">Qwen <span className="font-medium">/ chat lab</span></span>
        </div>
        <div className="eyebrow hidden sm:block">LOCAL INFERENCE WORKSPACE</div>
      </header>

      <section className="chat-frame" aria-label="Qwen chat conversation">
        <div className="chat-heading">
          <div>
            <div className="eyebrow">Adapter playground</div>
            <h1 className="heading-title">One conversation. Your local model.</h1>
          </div>
          <div className="server-pill" data-testid="status-server" aria-live="polite">
            <span className={`status-dot ${serverState}`} />
            <span>
              {serverState === 'checking' && 'Checking server'}
              {serverState === 'online' && 'Server online'}
              {serverState === 'offline' && 'Server unavailable'}
            </span>
          </div>
        </div>

        <div className="toolbar">
          <div className="toolbar-label" data-testid="text-conversation-count">
            <span className="h-1.5 w-1.5 rounded-full bg-primary" />
            {messages.length === 0 ? 'NEW SESSION' : `${messages.length} ${messages.length === 1 ? 'MESSAGE' : 'MESSAGES'}`}
          </div>
          <div className="mode-switch" role="group" aria-label="Answer mode">
            <button
              className={`mode-option ${mode === 'local' ? 'active' : ''}`}
              type="button"
              onClick={() => setMode('local')}
              disabled={isSending}
              aria-pressed={mode === 'local'}
              data-testid="button-mode-local"
            >
              <Bot size={13} />
              Local
            </button>
            <button
              className={`mode-option ${mode === 'research' ? 'active' : ''}`}
              type="button"
              onClick={() => setMode('research')}
              disabled={isSending}
              aria-pressed={mode === 'research'}
              data-testid="button-mode-research"
            >
              <Search size={13} />
              Research
            </button>
          </div>
          <button
            className="tool-button"
            type="button"
            onClick={startNewChat}
            disabled={isSending || messages.length === 0}
            data-testid="button-new-chat"
            aria-label="Start a new chat"
          >
            <Plus size={15} />
            New chat
          </button>
        </div>

        <section className="messages" aria-live="polite" aria-relevant="additions text" data-testid="list-messages">
          {messages.length === 0 ? (
            <div className="empty-state" data-testid="state-empty">
              <div className="empty-orbit" aria-hidden="true"><Sparkles size={25} strokeWidth={1.7} /></div>
              <div className="eyebrow mb-3">Ready when you are</div>
              <h2 className="empty-title">Give your adapter<br />something to think about.</h2>
              <p className="empty-description">
                Local Qwen is experimental. Research opens source pages and shows relevant passages; if a page cannot be read, it labels the search summary. Verify important claims.
              </p>
              <div className="suggestions" aria-label="Example prompts">
                {[
                  'What makes a useful system prompt?',
                  'Show the key steps for solving 18 × 24.',
                  'Research a recent AI development.',
                ].map((prompt, index) => (
                  <button
                    className="suggestion"
                    type="button"
                    key={prompt}
                    onClick={() => submitMessage(prompt)}
                    disabled={isSending}
                    data-testid={`button-suggestion-${index}`}
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div>
              {messages.map((message, index) => (
                <article
                  className={`message-row ${message.role}`}
                  key={`${message.role}-${index}`}
                  data-testid={`message-${message.role}-${index}`}
                >
                  <div className={`message-avatar ${message.role}`} aria-hidden="true">
                    {message.role === 'assistant' ? <Bot size={16} /> : <UserRound size={16} />}
                  </div>
                  <div className="min-w-0">
                    <div className="message-meta">
                      <span className="message-name">
                        {message.role === 'assistant'
                          ? message.sources?.length ? 'Source summary' : 'Qwen adapter'
                          : 'You'}
                      </span>
                      <span className="message-role">
                        {message.role === 'assistant'
                          ? message.sources?.length ? 'WEB RESEARCH' : 'LOCAL REASONING'
                          : 'PROMPT'}
                      </span>
                    </div>
                    <div className="message-content" data-testid={`text-message-content-${index}`}>{message.content}</div>
                    {message.role === 'assistant' && !!message.sources?.length && (
                      <div className="source-list" aria-label="Research sources">
                        <div className="source-list-heading"><Search size={12} /> WEB SOURCES</div>
                        {message.sources.map((source, sourceIndex) => (
                          <a
                            className="source-card"
                            href={source.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            key={`${source.url}-${sourceIndex}`}
                          >
                            <span className="source-title">
                              <span>{source.title}</span>
                              <ExternalLink size={12} aria-hidden="true" />
                            </span>
                            <span className="source-detail">
                              <span className="source-host">{new URL(source.url).hostname.replace(/^www\./, '')}</span>
                              <span className="source-type">
                                {source.contentType === 'page_excerpt' ? 'PAGE EXCERPT' : 'SEARCH SUMMARY'}
                              </span>
                            </span>
                            <span className="source-snippet">{source.snippet}</span>
                          </a>
                        ))}
                        <p className="source-disclaimer">Page excerpts are used when available; otherwise these are search summaries. Open sources to verify.</p>
                      </div>
                    )}
                  </div>
                </article>
              ))}
              {isSending && (
                <article className="message-row assistant" data-testid="state-sending">
                  <div className="message-avatar assistant" aria-hidden="true"><Bot size={16} /></div>
                  <div>
                    <div className="message-meta">
                      <span className="message-name">{mode === 'research' ? 'Source summary' : 'Qwen adapter'}</span>
                        <span className="message-role">{mode === 'research' ? 'READING SOURCES' : 'GENERATING'}</span>
                    </div>
                    <div className="typing-line" aria-label="Generating reply"><span /><span /><span /></div>
                  </div>
                </article>
              )}
              <div ref={bottomRef} />
            </div>
          )}
          {requestError && (
            <div className="error-banner" role="alert" data-testid="state-error">
              <div className="flex gap-2">
                <CircleAlert size={16} className="mt-0.5 shrink-0" />
                <span>{requestError}</span>
              </div>
              <button
                type="button"
                onClick={() => failedRequest && requestReply(failedRequest.history, failedRequest.mode)}
                data-testid="button-retry"
              >
                Retry
              </button>
            </div>
          )}
        </section>

        <div className="composer-wrap">
          <form className="composer" onSubmit={onSubmit} data-testid="form-chat">
            <label className="sr-only" htmlFor="chat-composer">
              {mode === 'research' ? 'Ask a web research question' : 'Message the Qwen adapter'}
            </label>
            <textarea
              id="chat-composer"
              ref={textAreaRef}
              rows={1}
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={onComposerKeyDown}
              placeholder={mode === 'research' ? 'Ask a question to research on the web…' : 'Message your local Qwen adapter…'}
              maxLength={4000}
              disabled={isSending}
              data-testid="input-message"
            />
            <button
              className="send-button"
              type="submit"
              disabled={!draft.trim() || isSending}
              aria-label={isSending ? 'Sending message' : 'Send message'}
              data-testid="button-send"
            >
              {isSending ? <LoaderCircle size={17} className="animate-spin" /> : <ArrowUp size={18} strokeWidth={2.4} />}
            </button>
          </form>
          <div className={`mode-caption ${mode === 'research' ? 'research' : ''}`}>
            {mode === 'research'
              ? 'Research extracts relevant page passages directly so Qwen cannot rewrite the facts. Search summaries are labeled.'
              : 'Local Qwen is experimental; verify its answers.'}
          </div>
          <div className="composer-foot">
            <span className="footer-note"><Check size={12} /> Conversation saved in this browser</span>
            <span>{draft.length}/4000 <span className="hidden sm:inline">· Enter to send</span></span>
          </div>
        </div>
      </section>
    </main>
  );
}

const queryClient = new QueryClient();

function Router() {
  return (
    <RoutedErrorBoundary>
      <Switch>
        <Route path="/" component={ChatHome} />
        <Route component={NotFound} />
      </Switch>
    </RoutedErrorBoundary>
  );
}

function RoutedErrorBoundary({ children }: { children: ReactNode }) {
  const [location] = useLocation();
  return <ErrorBoundary resetKey={location}>{children}</ErrorBoundary>;
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <WouterRouter base={import.meta.env.BASE_URL.replace(/\/$/, '')}>
          <Router />
        </WouterRouter>
        <Toaster />
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
