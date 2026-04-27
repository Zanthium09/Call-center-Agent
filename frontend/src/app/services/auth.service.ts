// ============================================================
// FILE PATH: src/app/services/auth.service.ts
// ============================================================
import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { BehaviorSubject, Observable, tap, throwError } from 'rxjs';
import { catchError, timeout } from 'rxjs/operators';

const STORAGE_KEY = 'tetra_agent_id';

export interface LoginResponse {
  agent_id: string;
  display_name: string | null;
  created: boolean;
}

@Injectable({ providedIn: 'root' })
export class AuthService {
  readonly api = 'http://localhost:8000';
  private headers = new HttpHeaders({ 'Content-Type': 'application/json' });

  private _agentId$ = new BehaviorSubject<string | null>(this._readStored());
  readonly agentId$ = this._agentId$.asObservable();

  constructor(private http: HttpClient) {}

  private _readStored(): string | null {
    try {
      return localStorage.getItem(STORAGE_KEY);
    } catch {
      return null;
    }
  }

  get currentAgentId(): string | null {
    return this._agentId$.value;
  }

  isAuthed(): boolean {
    return !!this._agentId$.value;
  }

  /** Throws if not authenticated. Use in components that require an ID. */
  requireAgentId(): string {
    const id = this._agentId$.value;
    if (!id) {
      throw new Error('Not authenticated');
    }
    return id;
  }

  login(rawAgentId: string): Observable<LoginResponse> {
    const trimmed = (rawAgentId ?? '').trim().toLowerCase();
    return this.http
      .post<LoginResponse>(
        `${this.api}/login`,
        { agent_id: trimmed },
        { headers: this.headers },
      )
      .pipe(
        timeout(8000),
        tap((res) => {
          try {
            localStorage.setItem(STORAGE_KEY, res.agent_id);
          } catch {}
          this._agentId$.next(res.agent_id);
        }),
        catchError((err) => throwError(() => err)),
      );
  }

  logout(): void {
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {}
    this._agentId$.next(null);
  }
}
