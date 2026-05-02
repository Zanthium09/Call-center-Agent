// ============================================================
// FILE PATH: src/app/components/upload-dataset/upload-progress.component.ts
// ============================================================
import { Component, OnInit, OnDestroy, NgZone, ChangeDetectorRef } from '@angular/core';
import { CommonModule, DatePipe } from '@angular/common';
import { ActivatedRoute, Router } from '@angular/router';
import { AuthService } from '../../services/auth.service';
import { UploadService, BatchDetail } from '../../services/upload.service';

@Component({
  selector: 'app-upload-progress',
  standalone: true,
  imports: [CommonModule, DatePipe],
  templateUrl: './upload-progress.component.html',
  styleUrls: ['./upload-progress.component.scss'],
})
export class UploadProgressComponent implements OnInit, OnDestroy {
  agentId = '';
  batchId = '';
  batch: BatchDetail | null = null;
  loading = true;
  error = '';
  cancelling = false;
  pollTimer: any = null;
  startedMs = 0;

  constructor(
    private auth: AuthService,
    private route: ActivatedRoute,
    private router: Router,
    private upload: UploadService,
    private zone: NgZone,
    private cdr: ChangeDetectorRef,
  ) {}

  ngOnInit(): void {
    this.agentId = this.auth.currentAgentId || '';
    if (!this.agentId) { this.router.navigate(['/login']); return; }
    this.batchId = this.route.snapshot.paramMap.get('batchId') || '';
    this.startedMs = Date.now();
    this._poll();
    this.pollTimer = setInterval(() => this._poll(), 2000);
  }

  ngOnDestroy(): void {
    if (this.pollTimer) clearInterval(this.pollTimer);
  }

  private _poll() {
    this.upload.getBatch(this.batchId).subscribe({
      next: (b) => {
        this.zone.run(() => {
          this.batch = b;
          this.loading = false;
          this.cdr.detectChanges();
          if (['done', 'error', 'cancelled'].includes(b.state)) {
            if (this.pollTimer) { clearInterval(this.pollTimer); this.pollTimer = null; }
          }
        });
      },
      error: (e) => {
        this.zone.run(() => {
          this.error = e?.error?.detail || `Failed to fetch batch (${e?.status ?? 'network'}).`;
          this.loading = false;
          this.cdr.detectChanges();
        });
      },
    });
  }

  percent(): number {
    if (!this.batch || !this.batch.total) return 0;
    return Math.min(100,
      Math.round(((this.batch.processed + this.batch.failed) * 100) / this.batch.total));
  }

  etaSeconds(): number | null {
    if (!this.batch || !this.batch.processed || this.batch.processed === 0) return null;
    const elapsed = (Date.now() - this.startedMs) / 1000;
    const rate = this.batch.processed / elapsed; // convs/sec
    if (rate <= 0) return null;
    const remaining = (this.batch.total - this.batch.processed - this.batch.failed) / rate;
    return Math.max(0, Math.round(remaining));
  }

  etaLabel(): string {
    const s = this.etaSeconds();
    if (s === null) return '—';
    if (s < 60) return `${s}s`;
    const m = Math.floor(s / 60);
    return m < 60 ? `${m}m ${s % 60}s` : `${Math.floor(m / 60)}h ${m % 60}m`;
  }

  cancel() {
    if (!this.batch || ['done', 'error', 'cancelled'].includes(this.batch.state)) return;
    this.cancelling = true;
    this.upload.cancelBatch(this.batchId).subscribe({
      next: () => this.zone.run(() => { this.cancelling = false; this.cdr.detectChanges(); }),
      error: () => this.zone.run(() => { this.cancelling = false; this.cdr.detectChanges(); }),
    });
  }

  goHistory() { this.router.navigate(['/history']); }
  goUpload() { this.router.navigate(['/upload']); }
  goChat() { this.router.navigate(['/chat']); }
  logout() { this.auth.logout(); this.router.navigate(['/login']); }
}
