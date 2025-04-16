// components/LastStepViewer.tsx
import React from 'react';
import { AgentStateResponse } from '../lib/api'; // Assuming type is defined here

interface LastStepViewerProps {
  lastStep: AgentStateResponse['last_step']; // Use the type from AgentStateResponse
}

const LastStepViewer: React.FC<LastStepViewerProps> = ({ lastStep }) => {
  if (!lastStep) {
    return (
      <div className="bg-white shadow rounded-lg p-4 mt-4">
        <h2 className="text-lg font-semibold text-gray-800 mb-2">Last Cognitive Step</h2>
        <p>No step information available yet.</p>
      </div>
    );
  }

  return (
    <div className="bg-white shadow rounded-lg p-4 mt-4">
      <h2 className="text-lg font-semibold text-gray-800 mb-2">Last Cognitive Step</h2>
      <div className="space-y-2">
        <div>
          <h3 className="font-medium text-gray-600">Thought:</h3>
          <p className="text-gray-800 whitespace-pre-wrap">{lastStep.thought || 'N/A'}</p>
        </div>
        <div>
          <h3 className="font-medium text-gray-600">Action:</h3>
          <p className="text-gray-800 font-mono bg-gray-100 p-1 rounded inline-block">{lastStep.action || 'N/A'}</p>
        </div>
        <div>
          <h3 className="font-medium text-gray-600">Action Input:</h3>
          <pre className="text-xs text-gray-700 bg-gray-100 p-2 rounded overflow-x-auto">
            {lastStep.action_input ? JSON.stringify(lastStep.action_input, null, 2) : 'N/A'}
          </pre>
        </div>
        <div>
          <h3 className="font-medium text-gray-600">Observation:</h3>
          <pre className="text-xs text-gray-700 bg-gray-100 p-2 rounded overflow-x-auto">
            {lastStep.observation ? JSON.stringify(lastStep.observation, null, 2) : 'N/A'}
          </pre>
        </div>
      </div>
    </div>
  );
};

export default LastStepViewer; 