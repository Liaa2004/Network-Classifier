// // API service for backend communication
// class ApiService {
//   constructor() {
//     this.baseURL = 'http://localhost:3001/api';
//   }

//   // Live capture endpoints
//   async startLiveCapture(config) {
//     const response = await fetch(`${this.baseURL}/sessions/start`, {
//       method: 'POST',
//       headers: {
//         'Content-Type': 'application/json',
//       },
//       body: JSON.stringify(config),
//     });
//     return response.json();
//   }

//   async stopLiveCapture(sessionId) {
//     const response = await fetch(`${this.baseURL}/sessions/${sessionId}/stop`, {
//       method: 'POST',
//     });
//     return response.json();
//   }

//   async getLiveFlows(sessionId, lastNSeconds = null) {
//     const params = new URLSearchParams();
//     if (lastNSeconds) {
//       params.append('lastNSeconds', lastNSeconds);
//     }
    
//     const response = await fetch(
//       `${this.baseURL}/flows/${sessionId}?${params}`
//     );
//     return response.json();
//   }

//   // File upload endpoints
//   async uploadPcapFile(file) {
//     const formData = new FormData();
//     formData.append('pcap', file);

//     const response = await fetch(`${this.baseURL}/upload`, {
//       method: 'POST',
//       body: formData,
//     });
//     return response.json();
//   }

//   async getUploadAnalysis(uploadId) {
//     const response = await fetch(`${this.baseURL}/analysis/${uploadId}`);
//     return response.json();
//   }

//   // History endpoints
//   async getSessionHistory() {
//     const response = await fetch(`${this.baseURL}/history`);
//     return response.json();
//   }

//   async deleteSession(sessionId) {
//     const response = await fetch(`${this.baseURL}/sessions/${sessionId}`, {
//       method: 'DELETE',
//     });
//     return response.json();
//   }

//   async downloadSession(sessionId) {
//     const response = await fetch(`${this.baseURL}/sessions/${sessionId}/download`);
//     return response.blob();
//   }
// }

// export const apiService = new ApiService();
// export default apiService;
// API service for backend communication
// frontend/src/services/api.ts

type RealtimeCallback = (data: any) => void;

interface UploadResult {
  success: boolean;
  uploadId?: string;
  filename?: string;
  [key: string]: any;
}

interface AnalysisResult {
  [key: string]: any;
}

interface Session {
  id: string;
  startTime: string;
  endTime?: string;
  [key: string]: any;
}

class ApiService {
  private baseURL: string;
  private wsUrl: string;
  private ws: WebSocket | null;
  private wsCallbacks: Map<string, RealtimeCallback>;

  constructor() {
    this.baseURL = 'https://network-classifier-2.onrender.com';
    this.wsUrl = 'wss://network-classifier-2.onrender.com';
    this.ws = null;
    this.wsCallbacks = new Map();
  }

  // WebSocket connection for real-time updates
  private connectWebSocket() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      return;
    }

    this.ws = new WebSocket(this.wsUrl);

    this.ws.onopen = () => {
      console.log('WebSocket connected');
    };

    this.ws.onmessage = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data);
        this.wsCallbacks.forEach(callback => callback(data));
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    this.ws.onclose = () => {
      console.log('WebSocket disconnected');
      // Reconnect after 3 seconds
      setTimeout(() => this.connectWebSocket(), 3000);
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }

  // Subscribe to real-time updates
  public onRealtimeUpdate(callback: RealtimeCallback): () => void {
    const id = `${Date.now()}-${Math.random()}`;
    this.wsCallbacks.set(id, callback);

    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      this.connectWebSocket();
    }

    return () => this.wsCallbacks.delete(id);
  }

  // Live capture endpoints
  public async startLiveCapture(config: Record<string, any>): Promise<any> {
    const response = await fetch(`${this.baseURL}/sessions/start`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(config),
    });
    const result = await response.json();

    this.connectWebSocket();
    return result;
  }

  public async stopLiveCapture(sessionId: string): Promise<any> {
    const response = await fetch(`${this.baseURL}/sessions/${sessionId}/stop`, {
      method: 'POST',
    });
    return response.json();
  }

  public async getLiveFlows(sessionId: string, lastNSeconds?: number): Promise<any> {
    const params = new URLSearchParams();
    if (lastNSeconds) {
      params.append('lastNSeconds', lastNSeconds.toString());
    }

    const response = await fetch(`${this.baseURL}/flows/${sessionId}?${params}`);
    return response.json();
  }

  public async getFlowStats(sessionId: string, lastNSeconds?: number): Promise<any> {
    const params = new URLSearchParams();
    if (lastNSeconds) {
      params.append('lastNSeconds', lastNSeconds.toString());
    }

    const response = await fetch(`${this.baseURL}/flows/${sessionId}/stats?${params}`);
    return response.json();
  }

  // File upload endpoints
 public async uploadPcapFile(file: File): Promise<any> {
  const formData = new FormData();
  formData.append('pcap', file);

  const response = await fetch(`${this.baseURL}/upload`, {
    method: 'POST',
    body: formData,
  });

  const uploadResult: UploadResult = await response.json();

  // Use the analysis already returned
  if (uploadResult.success && uploadResult.analysis) {
    return {
      ...uploadResult,
      flows: uploadResult.analysis.flows || [],
      stats: uploadResult.analysis.stats || {
        totalFlows: 0,
        webCount: 0,
        multimediaCount: 0,
        socialCount: 0,
        maliciousCount: 0
      }
    };
  }

  return uploadResult;
}


  public async getAnalysisResult(analysisId: string): Promise<AnalysisResult> {
    const response = await fetch(`${this.baseURL}/sessions/${analysisId}`);
    return response.json();
  }

  // History endpoints
  public async getSessionHistory(): Promise<Session[]> {
    const response = await fetch(`${this.baseURL}/sessions`);
    return response.json();
  }

  public async deleteSession(sessionId: string): Promise<any> {
    const response = await fetch(`${this.baseURL}/sessions/${sessionId}`, {
      method: 'DELETE',
    });
    return response.json();
  }

  public async getSession(sessionId: string): Promise<Session> {
    const response = await fetch(`${this.baseURL}/sessions/${sessionId}`);
    return response.json();
  }

  // Health check
  public async healthCheck(): Promise<any> {
    const response = await fetch(`${this.baseURL}/health`);
    return response.json();
  }
}

export const apiService = new ApiService();
export default apiService;
