// ============================================================
// FILE PATH: src/app/services/history.service.ts
// ============================================================
import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError, timeout } from 'rxjs/operators';

export interface SessionSummary {
  session_id: string;
  scenario_persona: string | null;
  scenario_issue: string | null;
  difficulty: number | null;
  outcome: 'win' | 'loss' | null;
  total_turns: number | null;
  average_score: number | null;
  trend: string | null;
  professionalism: number | null;
  customer_satisfaction: number | null;
  problem_resolution: number | null;
  empathy: number | null;
  communication_clarity: number | null;
  best_turn_score: number | null;
  worst_turn_score: number | null;
  created_via: string | null;
  parent_session_id: string | null;
  started_at: string | null;
  ended_at: string | null;
}

export interface HistoryResponse {
  agent_id: string;
  sessions: SessionSummary[];
}

export interface SessionFull extends SessionSummary {
  agent_id: string;
  best_turn_agent: string | null;
  worst_turn_agent: string | null;
  report_text: string | null;
  points: string[];
  turn_log: {
    turn: number;
    agent: string;
    customer: string;
    score: number;
    tip?: string;
  }[];
}

@Injectable({ providedIn: 'root' })
export class HistoryService {
  readonly api = 'http://localhost:8000';

  constructor(private http: HttpClient) {}

  getHistory(agentId: string): Observable<HistoryResponse> {
    return this.http
      .get<HistoryResponse>(`${this.api}/history/${encodeURIComponent(agentId)}`)
      .pipe(timeout(15000), catchError((err) => throwError(() => err)));
  }

  getSession(agentId: string, sessionId: string): Observable<SessionFull> {
    return this.http
      .get<SessionFull>(
        `${this.api}/history/${encodeURIComponent(agentId)}/${encodeURIComponent(sessionId)}`,
      )
      .pipe(timeout(15000), catchError((err) => throwError(() => err)));
  }
}
