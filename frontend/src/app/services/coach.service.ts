// ============================================================
// FILE PATH: src/app/services/coach.service.ts
// ============================================================
import { Injectable, OnDestroy } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError, timeout } from 'rxjs/operators';

export interface WsMessage {
  type: 'thinking' | 'result' | 'error' | 'pong' | 'edit_result' | 'redo' | 'ideals_update';   // ← ADDED ideals_update
  customer_reply?: string;
  score?: number;
  tip?: string;
  reason?: string;
  resolved?: boolean;
  ideal?: string;
  ideals?: { positive: string; neutral: string; negative: string; ideal: string };
  outcome?: string | null;
  customer_mood?: number;
  mood_label?: string;
  turns_remaining?: number;
  detail?: string;
  turn?: number;
  edited_turn?: number;    // ← ADDED for edit_result
  message?: string;        // ← ADDED for redo
  scenario?: any;          // ← ADDED for redo
}

@Injectable({ providedIn: 'root' })
export class CoachService implements OnDestroy {

  readonly api = 'http://localhost:8000';
  readonly wsBase = 'ws://localhost:8000';

  private jsonHeaders = new HttpHeaders({ 'Content-Type': 'application/json' });
  private socket: WebSocket | null = null;
  private pingTimer: any = null;
  private onMsgCb: ((msg: WsMessage) => void) | null = null;

  constructor(private http: HttpClient) { }

  ngOnDestroy() { this.closeSocket(); }

  connectSocket(sessionId: string, callback: (msg: WsMessage) => void): void {
    this.closeSocket();
    this.onMsgCb = callback;

    const url = `${this.wsBase}/ws/${sessionId}`;
    console.log('[WS] connecting to', url);
    this.socket = new WebSocket(url);

    this.socket.onopen = () => {
      console.log('[WS] open');
      this.pingTimer = setInterval(() => {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
          this.socket.send('ping');
        }
      }, 25000);
    };

    this.socket.onmessage = (evt: MessageEvent) => {
      try {
        const msg: WsMessage = JSON.parse(evt.data);
        console.log('[WS] received type=' + msg.type, msg);
        if (this.onMsgCb) {
          this.onMsgCb(msg);
        }
      } catch (e) {
        console.error('[WS] parse error', e);
      }
    };

    this.socket.onerror = (e) => {
      console.error('[WS] error', e);
    };

    this.socket.onclose = (e) => {
      console.log('[WS] closed code=' + e.code);
      clearInterval(this.pingTimer);
    };
  }

  closeSocket(): void {
    clearInterval(this.pingTimer);
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
    this.onMsgCb = null;
  }

  checkHealth(): Observable<any> {
    return this.http.get<any>(`${this.api}/health`)
      .pipe(timeout(8000), catchError(err => throwError(() => err)));
  }

  startSession(difficulty: number = 1): Observable<any> {
    return this.http.get<any>(`${this.api}/scenario?difficulty=${difficulty}`)
      .pipe(timeout(15000), catchError(err => throwError(() => err)));
  }

  sendMessage(sessionId: string, agentInput: string): Observable<any> {
    return this.http.post<any>(
      `${this.api}/message`,
      { session_id: sessionId, agent_input: agentInput },
      { headers: this.jsonHeaders }
    ).pipe(timeout(10000), catchError(err => throwError(() => err)));
  }

  getReport(sessionId: string): Observable<any> {
    return this.http.get<any>(`${this.api}/report/${sessionId}`)
      .pipe(timeout(120000), catchError(err => throwError(() => err)));
  }

  // ══════════════════════════════════════════════════════════════════════════
  //  NEW — Feature 1: LLM Scenario Generator
  // ══════════════════════════════════════════════════════════════════════════
  generateScenario(difficulty: number = 1): Observable<any> {
    return this.http.post<any>(
      `${this.api}/generate-scenario`,
      { difficulty },
      { headers: this.jsonHeaders }
    ).pipe(timeout(60000), catchError(err => throwError(() => err)));
  }

  // ══════════════════════════════════════════════════════════════════════════
  //  NEW — Feature 1b: Custom User-Defined Scenario
  // ══════════════════════════════════════════════════════════════════════════
  customScenario(issueType: string, persona: string, description: string, difficulty: number = 1): Observable<any> {
    return this.http.post<any>(
      `${this.api}/custom-scenario`,
      { issue_type: issueType, persona, description, difficulty },
      { headers: this.jsonHeaders }
    ).pipe(timeout(15000), catchError(err => throwError(() => err)));
  }

  // ══════════════════════════════════════════════════════════════════════════
  //  NEW — Feature 2: Edit Previous Agent Reply
  // ══════════════════════════════════════════════════════════════════════════
  editMessage(sessionId: string, turnNumber: number, newAgentInput: string): Observable<any> {
    return this.http.post<any>(
      `${this.api}/edit-message`,
      { session_id: sessionId, turn_number: turnNumber, new_agent_input: newAgentInput },
      { headers: this.jsonHeaders }
    ).pipe(timeout(60000), catchError(err => throwError(() => err)));
  }

  // ══════════════════════════════════════════════════════════════════════════
  //  NEW — Feature 3: Redo Conversation
  // ══════════════════════════════════════════════════════════════════════════
  redoConversation(sessionId: string): Observable<any> {
    return this.http.post<any>(
      `${this.api}/redo`,
      { session_id: sessionId },
      { headers: this.jsonHeaders }
    ).pipe(timeout(15000), catchError(err => throwError(() => err)));
  }
}