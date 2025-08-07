
import { useLiveDataStore } from "@/store/live-data";

const ConnectionStatus = () => {
  const { isConnected } = useLiveDataStore();

  return (
    <div className={`flex items-center gap-2 p-4 border rounded-lg`}>
      <div className={`p-2 rounded-full ${isConnected ? 'bg-green-100 dark:bg-green-900' : 'bg-red-100 dark:bg-red-900'}`}>
        <div className={`h-4 w-4 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
      </div>
      <div className="flex-1">
        <p className="font-medium">System Status</p>
        <p className="text-sm text-muted-foreground">
          {isConnected ? 'All systems operational - API connected' : 'Disconnected from server'}
        </p>
      </div>
      <div className={`h-2 w-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
    </div>
  );
};

export default ConnectionStatus;
