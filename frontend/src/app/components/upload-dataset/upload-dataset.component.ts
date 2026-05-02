// ============================================================
// FILE PATH: src/app/components/upload-dataset/upload-dataset.component.ts
// ============================================================
import { Component, NgZone, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { AuthService } from '../../services/auth.service';
import { UploadService } from '../../services/upload.service';

@Component({
  selector: 'app-upload-dataset',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './upload-dataset.component.html',
  styleUrls: ['./upload-dataset.component.scss'],
})
export class UploadDatasetComponent {
  agentId = '';
  selectedFile: File | null = null;
  defaultPersona = 'Unknown';
  defaultIssue = 'General Inquiry';
  dragActive = false;

  loading = false;
  error = '';

  readonly maxBytes = 10 * 1024 * 1024;
  readonly accepted = '.json,.jsonl,.csv,.txt,.log,.zip';

  constructor(
    private auth: AuthService,
    private upload: UploadService,
    private router: Router,
    private zone: NgZone,
    private cdr: ChangeDetectorRef,
  ) {
    this.agentId = auth.currentAgentId || '';
    if (!this.agentId) {
      this.router.navigate(['/login']);
    }
  }

  // ── Drag & drop ──
  onDragOver(e: DragEvent) { e.preventDefault(); this.dragActive = true; }
  onDragLeave(e: DragEvent) { e.preventDefault(); this.dragActive = false; }
  onDrop(e: DragEvent) {
    e.preventDefault();
    this.dragActive = false;
    if (e.dataTransfer?.files?.length) {
      this._setFile(e.dataTransfer.files[0]);
    }
  }
  onFilePicked(e: Event) {
    const f = (e.target as HTMLInputElement).files?.[0];
    if (f) this._setFile(f);
  }

  private _setFile(f: File) {
    if (f.size === 0) { this.error = 'File is empty.'; return; }
    if (f.size > this.maxBytes) {
      this.error = `File too large (${(f.size / 1024 / 1024).toFixed(1)} MB) — max 10 MB.`;
      return;
    }
    this.error = '';
    this.selectedFile = f;
  }

  fileSizeLabel(): string {
    if (!this.selectedFile) return '';
    const kb = this.selectedFile.size / 1024;
    return kb < 1024 ? `${kb.toFixed(1)} KB` : `${(kb / 1024).toFixed(2)} MB`;
  }

  // ── Submit ──
  startUpload() {
    if (!this.selectedFile) { this.error = 'Please pick a file first.'; return; }
    this.error = '';
    this.loading = true;
    this.upload.uploadDataset(
      this.selectedFile, this.agentId,
      this.defaultPersona || 'Unknown',
      this.defaultIssue || 'General Inquiry',
    ).subscribe({
      next: (res) => {
        this.zone.run(() => {
          this.loading = false;
          this.cdr.detectChanges();
          this.router.navigate(['/upload', res.batch_id]);
        });
      },
      error: (e) => {
        this.zone.run(() => {
          this.loading = false;
          this.error = e?.error?.detail || `Upload failed (${e?.status ?? 'network'}).`;
          this.cdr.detectChanges();
        });
      },
    });
  }

  goBack() { this.router.navigate(['/history']); }
  goChat() { this.router.navigate(['/chat']); }
  logout() { this.auth.logout(); this.router.navigate(['/login']); }
}
