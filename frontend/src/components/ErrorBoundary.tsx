import React from 'react';
import { log } from '@/utils/logger';

class ErrorBoundary extends React.Component<React.PropsWithChildren> {
  state = { hasError: false } as { hasError: boolean };

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    log('error', 'React render error', { error: error.stack ?? error.message, info });
  }

  render() {
    if ((this.state as { hasError: boolean }).hasError) {
      return (
        <div className="p-4 text-red-400">
          문제가 발생했습니다. 새로고침 해주세요.
        </div>
      );
    }
    return this.props.children;
  }
}

export default ErrorBoundary;
