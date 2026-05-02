// ============================================================
// FILE PATH: src/app/services/upload.service.ts
// ============================================================
import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError, timeout } from 'rxjs/operators';

export interface BatchSummary {
  batch_id: string;
  filename: string;
  format: string;
  total: number;
  processed: number;
  failed: number;
  state: 'pending' | 'running' | 'done' | 'error' | 'cancelled';
  error_message: string | null;
  started_at: string | null;
  finished_at: string | null;
}

export interface BatchDetail extends BatchSummary {
  failures: { id: string; reason: string }[];
  live: {
    current_conv?: string;
    current_conv_idx?: number;
    current_persona?: string;
    current_issue?: string;
    current_turns?: number;
  };
}

export interface AgentSummary {
  agent_id: string;
  total_sessions: number;
  live_sessions: number;
  uploaded_sessions: number;
  wins: number;
  losses: number;
  win_rate: number | null;
  lifetime_avg: number | null;
  lifetime_param_avgs: {
    professionalism: number | null;
    customer_satisfaction: number | null;
    problem_resolution: number | null;
    empathy: number | null;
    communication_clarity: number | null;
  };
  weakest_skill: string | null;
  strongest_skill: string | null;
}

@Injectable({ providedIn: 'root' })
export class UploadService {
  readonly api = 'http://localhost:8000';

  constructor(private http: HttpClient) {}

  uploadDataset(
    file: File, agentId: string,
    defaultPersona: string, defaultIssue: string,
  ): Observable<{ batch_id: string; state: string }> {
    const fd = new FormData();
    fd.append('file', file);
    fd.append('agent_id', agentId);
    fd.append('default_persona', defaultPersona);
    fd.append('default_issue', defaultIssue);
    return this.http.post<any>(`${this.api}/upload-dataset`, fd)
      .pipe(timeout(60000), catchError(err => throwError(() => err)));
  }

  getBatch(batchId: string): Observable<BatchDetail> {
    return this.http.get<BatchDetail>(`${this.api}/upload-batch/${batchId}`)
      .pipe(timeout(15000), catchError(err => throwError(() => err)));
  }

  cancelBatch(batchId: string): Observable<any> {
    return this.http.post<any>(`${this.api}/upload-batch/${batchId}/cancel`, {})
      .pipe(timeout(10000), catchError(err => throwError(() => err)));
  }

  deleteBatch(batchId: string, agentId: string): Observable<any> {
    return this.http.delete<any>(
      `${this.api}/upload-batch/${batchId}?agent_id=${encodeURIComponent(agentId)}`,
    ).pipe(timeout(10000), catchError(err => throwError(() => err)));
  }

  listBatches(agentId: string): Observable<{ batches: BatchSummary[] }> {
    return this.http.get<any>(
      `${this.api}/upload-batches/${encodeURIComponent(agentId)}`,
    ).pipe(timeout(15000), catchError(err => throwError(() => err)));
  }

  getAgentSummary(agentId: string): Observable<AgentSummary> {
    return this.http.get<AgentSummary>(
      `${this.api}/agent-summary/${encodeURIComponent(agentId)}`,
    ).pipe(timeout(15000), catchError(err => throwError(() => err)));
  }
}
