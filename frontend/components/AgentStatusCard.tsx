// components/AgentStatusCard.tsx

import React from 'react';

interface AgentStatusCardProps {
  status: string;
  activeFragment: string | null;
  currentTask: string | null;
}

const AgentStatusCard: React.FC<AgentStatusCardProps> = ({ status, activeFragment, currentTask }) => {
  return (
    <div className="bg-white shadow rounded-lg p-4 mb-4">
      <h2 className="text-lg font-semibold text-gray-800 mb-2">Agent Status</h2>
      <p><strong>Status:</strong> <span className={`font-medium ${status === 'error' ? 'text-red-600' : 'text-green-600'}`}>{status}</span></p>
      <p><strong>Active Fragment:</strong> {activeFragment || 'N/A'}</p>
      <p><strong>Current Task:</strong> {currentTask || 'Idle'}</p>
    </div>
  );
};

export default AgentStatusCard; 