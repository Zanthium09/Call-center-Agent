// ============================================================
// FILE PATH: src/app/components/history/history.component.ts
// ============================================================
import { Component, OnInit, NgZone, ChangeDetectorRef } from '@angular/core';
import { CommonModule, DatePipe } from '@angular/common';
import { Router, RouterLink } from '@angular/router';
import { BaseChartDirective } from 'ng2-charts';
import type { ChartConfiguration, ChartData, ChartType } from 'chart.js';
import { AuthService } from '../../services/auth.service';
import { HistoryService, SessionSummary } from '../../services/history.service';

@Component({
  selector: 'app-history',
  standalone: true,
  imports: [CommonModule, DatePipe, RouterLink, BaseChartDirective],
  templateUrl: './history.component.html',
  styleUrls: ['./history.component.scss'],
})
export class HistoryComponent implements OnInit {
  agentId = '';
  loading = true;
  error = '';
  sessions: SessionSummary[] = [];

  // ── Chart 1: average score per session over time ──
  avgChartData: ChartData<'line'> = { labels: [], datasets: [] };
  avgChartOptions: ChartConfiguration<'line'>['options'] = {
    responsive: true,
    maintainAspectRatio: false,
    scales: { y: { min: 0, max: 10, ticks: { stepSize: 2 } } },
    plugins: { legend: { display: false } },
  };

  // ── Chart 2: 5-parameter trend ──
  paramChartData: ChartData<'line'> = { labels: [], datasets: [] };
  paramChartOptions: ChartConfiguration<'line'>['options'] = {
    responsive: true,
    maintainAspectRatio: false,
    scales: { y: { min: 0, max: 10, ticks: { stepSize: 2 } } },
    plugins: { legend: { position: 'top', labels: { boxWidth: 12 } } },
  };

  // ── Chart 3: best vs worst turn per session ──
  bestWorstData: ChartData<'line'> = { labels: [], datasets: [] };
  bestWorstOptions: ChartConfiguration<'line'>['options'] = {
    responsive: true,
    maintainAspectRatio: false,
    scales: { y: { min: 0, max: 10, ticks: { stepSize: 2 } } },
    plugins: { legend: { position: 'top', labels: { boxWidth: 12 } } },
  };

  // ── Chart 4: win/loss outcome ──
  outcomeData: ChartData<'doughnut'> = { labels: [], datasets: [] };
  outcomeOptions: ChartConfiguration<'doughnut'>['options'] = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { position: 'bottom' } },
  };
  readonly chartType: ChartType = 'line';

  constructor(
    private auth: AuthService,
    private history: HistoryService,
    private router: Router,
    private zone: NgZone,
    private cdr: ChangeDetectorRef,
  ) {}

  ngOnInit(): void {
    const id = this.auth.currentAgentId;
    if (!id) {
      this.router.navigate(['/login']);
      return;
    }
    this.agentId = id;
    this._load();
  }

  private _load(): void {
    this.loading = true;
    this.error = '';
    this.history.getHistory(this.agentId).subscribe({
      next: (res) => {
        this.zone.run(() => {
          this.sessions = (res.sessions || []).slice().reverse();
          try { this._buildCharts(); } catch (chartErr) {
            console.error('[History] chart build error', chartErr);
          }
          this.loading = false;
          this.cdr.detectChanges();
        });
      },
      error: (e) => {
        this.zone.run(() => {
          this.error = e?.error?.detail || `Failed to load history (${e?.status ?? 'network'}).`;
          this.loading = false;
          this.cdr.detectChanges();
        });
      },
    });
  }

  private _buildCharts(): void {
    const labels = this.sessions.map((s, i) => `S${i + 1}`);

    // Chart 1 — avg score
    this.avgChartData = {
      labels,
      datasets: [
        {
          label: 'Average Score',
          data: this.sessions.map((s) => s.average_score ?? 0),
          borderColor: '#6366F1',
          backgroundColor: 'rgba(99,102,241,0.15)',
          tension: 0.3,
          fill: true,
          pointRadius: 4,
        },
      ],
    };

    // Chart 2 — 5 parameters
    const param = (k: keyof SessionSummary) =>
      this.sessions.map((s) => (s[k] as number | null) ?? 0);
    this.paramChartData = {
      labels,
      datasets: [
        { label: 'Professionalism', data: param('professionalism'),
          borderColor: '#3DBE6E', backgroundColor: 'rgba(61,190,110,.1)', tension: .3 },
        { label: 'Customer Satisfaction', data: param('customer_satisfaction'),
          borderColor: '#F59E0B', backgroundColor: 'rgba(245,158,11,.1)', tension: .3 },
        { label: 'Problem Resolution', data: param('problem_resolution'),
          borderColor: '#6366F1', backgroundColor: 'rgba(99,102,241,.1)', tension: .3 },
        { label: 'Empathy', data: param('empathy'),
          borderColor: '#EF4444', backgroundColor: 'rgba(239,68,68,.1)', tension: .3 },
        { label: 'Communication Clarity', data: param('communication_clarity'),
          borderColor: '#0EA5E9', backgroundColor: 'rgba(14,165,233,.1)', tension: .3 },
      ],
    };

    // Chart 3 — best vs worst
    this.bestWorstData = {
      labels,
      datasets: [
        {
          label: 'Best Turn',
          data: this.sessions.map((s) => s.best_turn_score ?? 0),
          borderColor: '#3DBE6E',
          backgroundColor: 'rgba(61,190,110,.1)',
          tension: 0.3,
        },
        {
          label: 'Worst Turn',
          data: this.sessions.map((s) => s.worst_turn_score ?? 0),
          borderColor: '#EF4444',
          backgroundColor: 'rgba(239,68,68,.1)',
          tension: 0.3,
        },
      ],
    };

    // Chart 4 — outcome doughnut
    const wins = this.sessions.filter((s) => s.outcome === 'win').length;
    const losses = this.sessions.filter((s) => s.outcome === 'loss').length;
    // Chart.js doughnut breaks when all values are 0; use a placeholder slice instead
    const hasOutcome = wins + losses > 0;
    this.outcomeData = {
      labels: hasOutcome ? ['Resolved', 'Failed'] : ['No data'],
      datasets: [
        {
          data: hasOutcome ? [wins, losses] : [1],
          backgroundColor: hasOutcome ? ['#3DBE6E', '#EF4444'] : ['#E2E8F0'],
          borderWidth: 0,
        },
      ],
    };
  }

  get totalSessions(): number {
    return this.sessions.length;
  }
  get totalWins(): number {
    return this.sessions.filter((s) => s.outcome === 'win').length;
  }
  get totalLosses(): number {
    return this.sessions.filter((s) => s.outcome === 'loss').length;
  }
  get overallAvg(): string {
    const arr = this.sessions
      .map((s) => s.average_score)
      .filter((v): v is number => v !== null && v !== undefined);
    if (!arr.length) return '—';
    return (arr.reduce((a, b) => a + b, 0) / arr.length).toFixed(1);
  }
  get bestEverScore(): number {
    return Math.max(0, ...this.sessions.map((s) => s.average_score ?? 0));
  }

  outcomeLabel(o: string | null): string {
    if (o === 'win') return '✅ Resolved';
    if (o === 'loss') return '❌ Failed';
    return '—';
  }

  scoreColor(score: number | null): string {
    if (score === null || score === undefined) return '#94A3B8';
    if (score >= 7) return '#3DBE6E';
    if (score >= 4) return '#F59E0B';
    return '#EF4444';
  }

  goLogin() {
    this.auth.logout();
    this.router.navigate(['/login']);
  }
  goChat() {
    this.router.navigate(['/chat']);
  }
}
