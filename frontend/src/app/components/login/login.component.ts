// ============================================================
// FILE PATH: src/app/components/login/login.component.ts
// ============================================================
import { Component, NgZone, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { AuthService } from '../../services/auth.service';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.scss'],
})
export class LoginComponent {
  agentIdInput = '';
  loading = false;
  error = '';
  greeting = '';

  constructor(
    private auth: AuthService,
    private router: Router,
    private zone: NgZone,
    private cdr: ChangeDetectorRef,
  ) {}

  private _validate(): string | null {
    const v = this.agentIdInput.trim().toLowerCase();
    if (!v) return 'Agent ID is required.';
    if (v.length > 64) return 'Agent ID is too long (max 64 chars).';
    if (!/^[a-z0-9._\-]+$/.test(v)) {
      return 'Use only letters, digits, dot, underscore, dash.';
    }
    return null;
  }

  continueToChat() {
    const err = this._validate();
    if (err) { this.error = err; return; }
    this.error = '';
    this.loading = true;
    this.auth.login(this.agentIdInput).subscribe({
      next: (res) => {
        this.zone.run(() => {
          this.loading = false;
          this.greeting = res.created
            ? `New agent "${res.agent_id}" created.`
            : `Welcome back, ${res.agent_id}.`;
          this.cdr.detectChanges();
          this.router.navigate(['/chat']);
        });
      },
      error: (e) => {
        this.zone.run(() => {
          this.loading = false;
          this.error = e?.error?.detail || `Login failed (${e?.status ?? 'network error'}).`;
          this.cdr.detectChanges();
        });
      },
    });
  }

  reviewPrevious() {
    const err = this._validate();
    if (err) { this.error = err; return; }
    this.error = '';
    this.loading = true;
    this.auth.login(this.agentIdInput).subscribe({
      next: () => {
        this.zone.run(() => {
          this.loading = false;
          this.cdr.detectChanges();
          this.router.navigate(['/history']);
        });
      },
      error: (e) => {
        this.zone.run(() => {
          this.loading = false;
          this.error = e?.error?.detail || `Login failed (${e?.status ?? 'network error'}).`;
          this.cdr.detectChanges();
        });
      },
    });
  }
}
