// components/LogViewer.tsx
import React from 'react';
import { LogEvent } from '../lib/api'; // Assuming LogEvent type is defined in api.ts

interface LogViewerProps {
  logs: LogEvent[];
  isLoading: boolean;
  error: Error | null;
}

const LogViewer: React.FC<LogViewerProps> = ({ logs, isLoading, error }) => {
  if (isLoading) return <p>Loading logs...</p>;
  if (error) return <p className="text-red-500">Error loading logs: {error.message}</p>;

  const getLogLevelColor = (level: string) => {
    switch (level.toUpperCase()) {
      case 'ERROR': return 'text-red-600';
      case 'WARNING': return 'text-yellow-600';
      case 'INFO': return 'text-blue-600';
      case 'DEBUG': return 'text-gray-500';
      default: return 'text-gray-700';
    }
  };

  return (
    <div className="bg-gray-50 shadow rounded-lg p-4 h-96 overflow-y-auto font-mono text-sm">
      <h2 className="text-lg font-semibold text-gray-800 mb-2 sticky top-0 bg-gray-50 py-1">Recent Logs</h2>
      {logs.length === 0 ? (
        <p>No logs available.</p>
      ) : (
        logs.map((log, index) => (
          <div key={index} className="border-b border-gray-200 py-1 flex">
            <span className="text-gray-400 mr-2 flex-shrink-0">{new Date(log.timestamp).toLocaleTimeString()}</span>
            <span className={`font-bold mr-2 flex-shrink-0 ${getLogLevelColor(log.level)}`}>[{log.level}]</span>
            {log.source && <span className="text-purple-600 mr-2 flex-shrink-0">({log.source})</span>}
            <span className="whitespace-pre-wrap break-words">{log.message}</span>
            {/* Optionally display extra_data */} 
            {/* {log.extra_data && <pre className="text-xs text-gray-400 mt-1 ml-4">{JSON.stringify(log.extra_data, null, 2)}</pre>} */} 
          </div>
        ))
      )}
    </div>
  );
};

export default LogViewer; 