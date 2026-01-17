type LogLevel = 'info' | 'warn' | 'error';

interface LogEntry {
  level: LogLevel;
  message: string;
  data?: unknown;
  time: string;
}

const STORAGE_KEY = 'appLogs';

function readStoredLogs(): LogEntry[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function writeStoredLogs(logs: LogEntry[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(logs.slice(-500)));
  } catch {
    // ignore storage errors
  }
}

function appendLog(entry: LogEntry) {
  const logs = readStoredLogs();
  logs.push(entry);
  writeStoredLogs(logs);
}

export function getLogs(): LogEntry[] {
  return readStoredLogs();
}

export function clearLogs() {
  localStorage.removeItem(STORAGE_KEY);
}

export function log(level: LogLevel, message: string, data?: unknown) {
  const time = new Date().toISOString();
  const entry: LogEntry = { level, message, data, time };
  appendLog(entry);
  // Mirror to console
  if (level === 'info') console.log(`[INFO ${time}] ${message}`, data ?? '');
  else if (level === 'warn') console.warn(`[WARN ${time}] ${message}`, data ?? '');
  else console.error(`[ERROR ${time}] ${message}`, data ?? '');
}

export function initLogger() {
  // Environment info
  log('info', 'Logger initialized', {
    url: window.location.href,
    userAgent: navigator.userAgent,
    language: navigator.language,
  });

  // Hook global errors
  window.addEventListener('error', (event) => {
    log('error', 'Global error', {
      message: event.message,
      filename: (event as ErrorEvent).filename,
      lineno: (event as ErrorEvent).lineno,
      colno: (event as ErrorEvent).colno,
      error: (event as ErrorEvent).error?.stack ?? (event as ErrorEvent).error,
    });
  });

  window.addEventListener('unhandledrejection', (event) => {
    log('error', 'Unhandled promise rejection', {
      reason: (event as PromiseRejectionEvent).reason,
    });
  });

  // Wrap console to also persist warnings/errors
  const originalError = console.error;
  console.error = (...args: unknown[]) => {
    try {
      log('error', 'Console error', { args });
    } catch {}
    originalError.apply(console, args as []);
  };

  const originalWarn = console.warn;
  console.warn = (...args: unknown[]) => {
    try {
      log('warn', 'Console warn', { args });
    } catch {}
    originalWarn.apply(console, args as []);
  };
}
