
import { toast } from 'sonner';

import useLiveDataStore from '../store/live-data';

class WebSocketService {
  private socket: WebSocket | null = null;
  private url: string;
  private reconnectAttempts = 0;

  constructor(url: string) {
    this.url = url;
  }

  connect() {
    this.socket = new WebSocket(this.url);

    this.socket.onopen = () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
    };

    this.socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        const { type, data: payload } = data;

        if (type === 'tick') {
          useLiveDataStore.getState().setLatestTick(payload);
        } else if (type === 'position_update') {
          useLiveDataStore.getState().setCurrentPosition(payload);
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    this.socket.onclose = () => {
      console.log('WebSocket disconnected');
      this.reconnect();
    };

    this.socket.onerror = (error) => {
      console.error('WebSocket error:', error);
      toast.error('WebSocket error. See console for details.');
    };
  }

  disconnect() {
    if (this.socket) {
      this.socket.close();
    }
  }

  private reconnect() {
    const delay = Math.min(30000, Math.pow(2, this.reconnectAttempts) * 1000);
    setTimeout(() => {
      this.reconnectAttempts++;
      this.connect();
    }, delay);
  }

  onMessage(callback: (data: any) => void) {
    if (this.socket) {
      this.socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          callback(data);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };
    }
  }
}

export default WebSocketService;
