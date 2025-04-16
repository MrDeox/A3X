'use client'; // Required for useState and useEffect

import React, { useState, useEffect, useCallback, useRef } from 'react';
import AgentStatusCard from '../components/AgentStatusCard';
import LogViewer from '../components/LogViewer';
import LastStepViewer from '../components/LastStepViewer';
import { fetchAgentState, fetchLogs, AgentStateResponse, LogEvent } from '../lib/api';

const POLLING_INTERVAL_MS = 3000; // Refresh every 3 seconds
const LOG_LIMIT = 50; // Number of logs to fetch

// --- Task Submission Form Component ---
// interface TaskSubmitFormProps {
//   onSubmit: (task: string) => Promise<void>;
//   disabled: boolean;
// }
//
// function TaskSubmitForm({ onSubmit, disabled }: TaskSubmitFormProps) {
//   const [taskInput, setTaskInput] = useState('');
//
//   const handleSubmit = async (event: React.FormEvent) => {
//     event.preventDefault();
//     if (!taskInput.trim()) return; // Don't submit empty tasks
//     await onSubmit(taskInput);
//     setTaskInput(''); // Clear input after submission
//   };
//
//   return (
//     <form onSubmit={handleSubmit} className="mt-6 mb-4 p-4 bg-white rounded-lg shadow">
//       <label htmlFor="taskInput" className="block text-sm font-medium text-gray-700 mb-1">
//         Submit New Task:
//       </label>
//       <div className="flex">
//         <input
//           type="text"
//           id="taskInput"
//           value={taskInput}
//           onChange={(e) => setTaskInput(e.target.value)}
//           placeholder="Enter task for the agent..."
//           className="flex-grow p-2 border border-gray-300 rounded-l-md focus:ring-indigo-500 focus:border-indigo-500"
//           disabled={disabled}
//         />
//         <button
//           type="submit"
//           className={`px-4 py-2 border border-transparent rounded-r-md shadow-sm text-sm font-medium text-white ${disabled ? 'bg-gray-400 cursor-not-allowed' : 'bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500'}`}
//           disabled={disabled}
//         >
//           Run Task
//         </button>
//       </div>
//     </form>
//   );
// }
// --- End Task Submission Form Component ---

export default function DashboardPage() {
  const [agentState, setAgentState] = useState<AgentStateResponse | null>(null);
  const [logs, setLogs] = useState<LogEvent[]>([]);
  const [isLoadingState, setIsLoadingState] = useState<boolean>(true);
  const [isLoadingLogs, setIsLoadingLogs] = useState<boolean>(true);
  const [stateError, setStateError] = useState<Error | null>(null);
  const [logError, setLogError] = useState<Error | null>(null);
  // const [isSubmittingTask, setIsSubmittingTask] = useState(false);

  // Ref to track if polling is active to prevent multiple intervals
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const fetchData = useCallback(async () => {
    // Fetch Agent State
    try {
      // Only set loading true on initial load, not subsequent polls
      // setStateError(null); // Clear previous error
      const stateData = await fetchAgentState();
      setAgentState(stateData);
    } catch (error) {
      console.error("Error fetching agent state:", error);
      setStateError(error instanceof Error ? error : new Error('Failed to fetch state'));
    } finally {
      // Only set loading false after initial load
      if (isLoadingState) setIsLoadingState(false);
    }

    // Fetch Logs
    try {
      // setLogError(null); // Clear previous error
      const logData = await fetchLogs(LOG_LIMIT);
      setLogs(logData.events); // Assuming the API returns { events: [...] }
    } catch (error) {
      console.error("Error fetching logs:", error);
      setLogError(error instanceof Error ? error : new Error('Failed to fetch logs'));
    } finally {
       // Only set loading false after initial load
      if (isLoadingLogs) setIsLoadingLogs(false);
    }
  }, []);

  // Function to handle task submission
  // const handleTaskSubmit = async (task: string) => {
  //   setIsSubmittingTask(true);
  //   setStateError(null); // Clear previous errors
  //   try {
  //     await submitAgentTask(task);
  //     // Optionally trigger an immediate data refresh after submission
  //     // Or just wait for the next polling cycle
  //     await fetchData(); // Refresh data immediately to show submitted status if backend updates state
  //   } catch (error) {
  //     console.error("Error submitting task:", error);
  //     // Display error to the user, maybe in a dedicated error area
  //     setStateError(error instanceof Error ? error : new Error('Failed to submit task'));
  //   } finally {
  //     setIsSubmittingTask(false);
  //   }
  // };

  useEffect(() => {
    // Fetch data immediately on mount
    fetchData();

    // Set up polling interval
    if (!pollingIntervalRef.current) {
      pollingIntervalRef.current = setInterval(fetchData, POLLING_INTERVAL_MS);
    }

    // Clean up interval on component unmount
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fetchData]); // Empty dependency array means this runs once on mount and cleans up on unmount

  // Determine if the agent is currently busy (useful for disabling submit button)
  // Refined logic: Only disable if submitting OR if state exists and status is not an idle/finished state.
  // const isAgentBusy = isSubmittingTask || (agentState != null && !['Idle', 'Completed', 'Error', 'unknown'].includes(agentState.status));

  // DEBUGGING:
  // useEffect(() => {
  //   console.log("Agent State Updated:", agentState);
  //   console.log("Is Submitting Task:", isSubmittingTask);
  //   console.log("Calculated isAgentBusy:", isAgentBusy);
  // }, [agentState, isSubmittingTask, isAgentBusy]);

  return (
    <main className="flex min-h-screen flex-col items-center justify-start p-6 md:p-12 bg-gray-100">
      <div className="w-full max-w-5xl">
        <h1 className="text-3xl font-bold text-center mb-6 text-gray-800">A³X Cognitive Dashboard</h1>

        {/* Task Submission Form - Removed */}
        {/* <TaskSubmitForm onSubmit={handleTaskSubmit} disabled={isAgentBusy} /> */}

        {isLoadingState ? (
          <p>Loading agent state...</p>
        ) : stateError ? (
          <p className="text-red-500">Error loading state: {stateError.message}</p>
        ) : agentState ? (
          <>
            <AgentStatusCard 
              status={agentState.status || 'unknown'} 
              activeFragment={agentState.active_fragment} 
              currentTask={agentState.current_task} 
            />
            <LastStepViewer lastStep={agentState.last_step} />
          </>
        ) : (
          <p>No agent state available.</p>
        )}

        <div className="mt-6">
          <LogViewer logs={logs} isLoading={isLoadingLogs} error={logError} />
        </div>

      </div>
    </main>
  );
}
