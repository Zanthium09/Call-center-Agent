// ============================================================
// FILE PATH: src/app/components/session-report/session-report.component.ts
// ============================================================
import { Component, OnInit, NgZone, ChangeDetectorRef } from '@angular/core';
import { CommonModule, DatePipe } from '@angular/common';
import { ActivatedRoute, Router, RouterLink } from '@angular/router';
import { AuthService } from '../../services/auth.service';
import { HistoryService, SessionFull } from '../../services/history.service';

@Component({
  selector: 'app-session-report',
  standalone: true,
  imports: [CommonModule, DatePipe, RouterLink],
  templateUrl: './session-report.component.html',
  styleUrls: ['./session-report.component.scss'],
})
export class SessionReportComponent implements OnInit {
  loading = true;
  error = '';
  data: SessionFull | null = null;

  constructor(
    private auth: AuthService,
    private route: ActivatedRoute,
    private router: Router,
    private history: HistoryService,
    private zone: NgZone,
    private cdr: ChangeDetectorRef,
  ) {}

  ngOnInit(): void {
    const id = this.auth.currentAgentId;
    if (!id) {
      this.router.navigate(['/login']);
      return;
    }
    const sessionId = this.route.snapshot.paramMap.get('sessionId') || '';
    this.history.getSession(id, sessionId).subscribe({
      next: (res) => {
        this.zone.run(() => {
          this.data = res;
          this.loading = false;
          this.cdr.detectChanges();
        });
      },
      error: (e) => {
        this.zone.run(() => {
          this.error =
            e?.error?.detail || `Failed to load session (${e?.status ?? 'network'}).`;
          this.loading = false;
          this.cdr.detectChanges();
        });
      },
    });
  }

  paramRows(): { label: string; value: number }[] {
    const d = this.data;
    return [
      { label: '👔 Professionalism', value: d?.professionalism ?? 0 },
      { label: '😊 Customer Satisfaction', value: d?.customer_satisfaction ?? 0 },
      { label: '🛠️ Problem Resolution', value: d?.problem_resolution ?? 0 },
      { label: '❤️ Empathy', value: d?.empathy ?? 0 },
      { label: '💬 Communication Clarity', value: d?.communication_clarity ?? 0 },
    ];
  }

  scoreClass(score: number | null | undefined): string {
    if (score === null || score === undefined) return 'low';
    if (score >= 7) return 'high';
    if (score >= 4) return 'mid';
    return 'low';
  }

  outcomeBadge(o: string | null): { label: string; cls: string } {
    if (o === 'win') return { label: '✅ Resolved', cls: 'high' };
    if (o === 'loss') return { label: '❌ Failed', cls: 'low' };
    return { label: '—', cls: 'mid' };
  }
}
