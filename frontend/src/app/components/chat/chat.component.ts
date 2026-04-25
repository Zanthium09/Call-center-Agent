// FILE PATH: src/app/components/chat/chat.component.ts
import {
  Component, OnInit, AfterViewChecked, OnDestroy,
  ElementRef, ViewChild, ChangeDetectorRef, NgZone
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { CoachService, WsMessage } from '../../services/coach.service';

interface Ideals { positive: string; neutral: string; negative: string; ideal: string; }
interface Message {
  role: 'agent' | 'customer'; text: string;
  score?: number; tip?: string; reason?: string;
  ideal?: string; ideals?: Ideals;
  mood?: number; moodLabel?: string; turnsLeft?: number;
  editing?: boolean; editText?: string;
}

@Component({
  selector: 'app-chat', standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.scss']
})
export class ChatComponent implements OnInit, AfterViewChecked, OnDestroy {
  @ViewChild('scrollContainer') scrollContainer!: ElementRef;

  sessionId = ''; scenario: any = null;
  messages: Message[] = []; agentInput = '';
  resolved = false; failed = false; loading = false;
  error = ''; logs: string[] = [];
  report: any = null; reportLoading = false; showReport = false;
  currentMood = 5; currentMoodLabel = 'Frustrated'; turnsRemaining = 20;
  selectedDifficulty = 1;
  readonly difficultyLevels = [1, 2, 3, 4, 5];
  processingLabel = '';

  // ── Feature 1 — Scenario Generator state ──
  generatingScenario = false;
  generatedDescription = '';

  // ── Feature 1b — Custom Scenario state ──
  customIssue = '';
  customPersona = '';
  customDescription = '';
  showCustomForm = false;

  constructor(
    private coach: CoachService,
    private cdr: ChangeDetectorRef,
    private zone: NgZone
  ) { }

  ngOnInit() { this.addLog('🟢 App started'); this.startNewSession(); }
  ngOnDestroy() { this.coach.closeSocket(); }
  ngAfterViewChecked() {
    if (this.scrollContainer)
      this.scrollContainer.nativeElement.scrollTop = this.scrollContainer.nativeElement.scrollHeight;
  }

  setDifficulty(level: number) {
    if (this.loading) return;
    this.selectedDifficulty = level;
    this.coach.closeSocket();
    this.messages = []; this.resolved = false; this.failed = false;
    this.error = ''; this.report = null; this.reportLoading = false;
    this.showReport = false; this.failed = false;
    this.generatedDescription = '';
    this.startNewSession();
  }

  difficultyLabel(level: number): string {
    return ['', 'Easy', 'Medium', 'Hard', 'Expert', 'Nightmare'][level] ?? '';
  }

  moodClass(mood: number): string {
    if (mood >= 9) return 'mood-satisfied'; if (mood >= 7) return 'mood-calming';
    if (mood >= 5) return 'mood-frustrated'; if (mood >= 3) return 'mood-angry';
    return 'mood-furious';
  }
  moodBarWidth(mood: number): number { return Math.round((mood / 10) * 100); }
  scoreClass(score: number): string { return score >= 7 ? 'high' : score >= 4 ? 'mid' : 'low'; }

  addLog(msg: string) {
    const line = `[${new Date().toLocaleTimeString()}] ${msg}`;
    this.logs.push(line);
    if (this.logs.length > 60) this.logs.shift();
    console.log(line);
  }
  clearLogs() { this.logs = []; this.cdr.detectChanges(); }

  // ── Mood helpers ──
  _moodLabelFor(mood: number): string {
    if (mood >= 9) return 'Satisfied';
    if (mood >= 7) return 'Calming';
    if (mood >= 5) return 'Frustrated';
    if (mood >= 3) return 'Angry';
    return 'Furious';
  }

  _setStartingMood(backendMood?: number) {
    this.currentMood = backendMood ?? 5;
    this.currentMoodLabel = this._moodLabelFor(this.currentMood);
  }

  startNewSession() {
    this.loading = true; this.error = '';
    this.addLog(`📡 Connecting (difficulty ${this.selectedDifficulty})...`);
    this.cdr.detectChanges();

    this.coach.checkHealth().subscribe({
      next: (h) => {
        this.addLog(`✅ Backend OK — LM Studio: ${h?.lm_studio ?? 'unknown'}`);
        this.coach.startSession(this.selectedDifficulty).subscribe({
          next: (res) => {
            this.sessionId = res.session_id; this.scenario = res.scenario;
            this._setStartingMood(res.starting_mood); this.turnsRemaining = 20;
            this.loading = false;
            this.addLog(`✅ ${res.scenario?.customer_persona} | ${res.scenario?.issue_type}`);
            this.cdr.detectChanges();
            this._connectWebSocket(res.session_id);
          },
          error: (err) => {
            this.addLog(`❌ /scenario failed — ${err.status}`);
            this.error = `Cannot load scenario (${err.status})`; this.loading = false;
            this.cdr.detectChanges();
          }
        });
      },
      error: (err) => {
        const msg = err.status === 0 ? 'Cannot reach backend — is uvicorn running?' : `Backend error ${err.status}`;
        this.addLog(`❌ ${msg}`); this.error = msg; this.loading = false;
        this.cdr.detectChanges();
      }
    });
  }

  private _connectWebSocket(sessionId: string): void {
    this.coach.connectSocket(sessionId, (msg: WsMessage) => {
      this.zone.run(() => {
        console.log('[Component] WS msg:', msg.type, JSON.stringify(msg).slice(0, 120));
        this._handle(msg);
      });
    });
    this.addLog('🔌 WebSocket connected');
    this.cdr.detectChanges();
  }

  private _handle(msg: WsMessage): void {
    switch (msg.type) {

      case 'thinking':
        this.processingLabel = 'Generating response...';
        this.cdr.detectChanges();
        break;

      case 'result':
        this.loading = false;
        this.processingLabel = '';
        this.currentMood = msg.customer_mood ?? this.currentMood;
        this.currentMoodLabel = msg.mood_label ?? this.currentMoodLabel;
        this.turnsRemaining = msg.turns_remaining ?? this.turnsRemaining;

        this.addLog(`✅ Score: ${msg.score}/10 | Mood: ${msg.mood_label} (${msg.customer_mood}/10)`);

        this.messages.push({
          role: 'customer', text: msg.customer_reply ?? '',
          score: msg.score, tip: msg.tip, reason: msg.reason,
          ideal: msg.ideal, ideals: msg.ideals as Ideals,
          mood: msg.customer_mood, moodLabel: msg.mood_label, turnsLeft: msg.turns_remaining,
        });

        if (msg.resolved || msg.outcome === 'win') {
          this.resolved = true;
          this.addLog('🏆 Issue resolved!');
          this.reportLoading = true;
          this.cdr.detectChanges();
          this.coach.getReport(this.sessionId).subscribe({
            next: r => this.zone.run(() => {
              this.report = r.report; this.reportLoading = false;
              this.addLog(`📊 Avg ${r.report?.average_score}/10`); this.cdr.detectChanges();
            }),
            error: () => this.zone.run(() => { this.reportLoading = false; this.cdr.detectChanges(); })
          });
        }

        if (msg.outcome === 'loss') {
          this.addLog('💀 Session failed.');
          this.failed = true;
          this.reportLoading = true;
          this.cdr.detectChanges();
          this.coach.getReport(this.sessionId).subscribe({
            next: r => this.zone.run(() => {
              this.report = r.report; this.reportLoading = false;
              this.addLog(`📊 Avg ${r.report?.average_score}/10`); this.cdr.detectChanges();
            }),
            error: () => this.zone.run(() => { this.reportLoading = false; this.cdr.detectChanges(); })
          });
        }
        this.cdr.detectChanges();
        break;

      case 'edit_result': {
        this.loading = false;
        this.processingLabel = '';
        const editedTurn = msg.edited_turn ?? 1;
        const agentIdx = (editedTurn - 1) * 2;

        this.addLog(`✏️ Edit T${editedTurn} — Score: ${msg.score}/10 | Mood: ${msg.mood_label}`);

        this.messages = this.messages.slice(0, agentIdx + 1);
        if (this.messages[agentIdx]) {
          this.messages[agentIdx].editing = false;
          this.messages[agentIdx].editText = undefined;
        }

        this.messages.push({
          role: 'customer', text: msg.customer_reply ?? '',
          score: msg.score, tip: msg.tip, reason: msg.reason,
          ideal: msg.ideal, ideals: msg.ideals as Ideals,
          mood: msg.customer_mood, moodLabel: msg.mood_label, turnsLeft: msg.turns_remaining,
        });

        this.currentMood = msg.customer_mood ?? this.currentMood;
        this.currentMoodLabel = msg.mood_label ?? this.currentMoodLabel;
        this.turnsRemaining = msg.turns_remaining ?? this.turnsRemaining;

        if (msg.resolved || msg.outcome === 'win') {
          this.resolved = true;
          this.addLog('🏆 Issue resolved (after edit)!');
          this.reportLoading = true;
          this.cdr.detectChanges();
          this.coach.getReport(this.sessionId).subscribe({
            next: r => this.zone.run(() => {
              this.report = r.report; this.reportLoading = false; this.cdr.detectChanges();
            }),
            error: () => this.zone.run(() => { this.reportLoading = false; this.cdr.detectChanges(); })
          });
        }
        if (msg.outcome === 'loss') {
          this.failed = true;
          this.addLog('💀 Session failed (after edit).');
          this.reportLoading = true;
          this.cdr.detectChanges();
          this.coach.getReport(this.sessionId).subscribe({
            next: r => this.zone.run(() => {
              this.report = r.report; this.reportLoading = false; this.cdr.detectChanges();
            }),
            error: () => this.zone.run(() => { this.reportLoading = false; this.cdr.detectChanges(); })
          });
        }
        this.cdr.detectChanges();
        break;
      }

      case 'redo':
        this.addLog('🔄 Conversation restarted (same scenario)');
        this.cdr.detectChanges();
        break;

      // ══════════════════════════════════════════════════════════════════════
      //  NEW — Handle ideals_update: patch ideals into the last customer msg
      // ══════════════════════════════════════════════════════════════════════
      case 'ideals_update': {
        // Find the last customer message and patch its ideals
        for (let i = this.messages.length - 1; i >= 0; i--) {
          if (this.messages[i].role === 'customer') {
            this.messages[i].ideal = msg.ideal;
            this.messages[i].ideals = msg.ideals as Ideals;
            break;
          }
        }
        this.addLog('🌟 Ideal responses loaded');
        this.cdr.detectChanges();
        break;
      }

      case 'error':
        this.loading = false; this.processingLabel = '';
        this.addLog(`❌ Error: ${msg.detail}`);
        this.error = 'Backend error — check LM Studio is running';
        this.cdr.detectChanges();
        break;
    }
  }

  send() {
    if (!this.agentInput.trim() || this.loading) return;
    const input = this.agentInput.trim();
    this.agentInput = ''; this.loading = true; this.error = '';
    this.processingLabel = 'Sending...';
    this.messages.push({ role: 'agent', text: input });
    this.cdr.detectChanges();

    this.coach.sendMessage(this.sessionId, input).subscribe({
      next: () => { this.processingLabel = 'Analysing your reply...'; this.cdr.detectChanges(); },
      error: (err) => {
        this.addLog(`❌ /message failed — ${err.status}`);
        this.error = `Error (${err.status})`; this.loading = false;
        this.processingLabel = ''; this.cdr.detectChanges();
      }
    });
  }

  newSession() {
    this.coach.closeSocket();
    this.loading = false;
    this.messages = []; this.resolved = false; this.failed = false;
    this.scenario = null; this.error = ''; this.report = null;
    this.reportLoading = false; this.showReport = false; this.currentMood = 5;
    this.currentMoodLabel = 'Frustrated'; this.turnsRemaining = 20;
    this.processingLabel = '';
    this.generatedDescription = '';
    this.generatingScenario = false;
    this.customIssue = ''; this.customPersona = ''; this.customDescription = '';
    this.showCustomForm = false;
    this.cdr.detectChanges();
    this.startNewSession();
  }

  // ════════════════════════════════════════════════════════════════════════════
  //  Feature 1: Generate a random scenario via LLM
  // ════════════════════════════════════════════════════════════════════════════
  generateScenario() {
    if (this.generatingScenario || this.loading) return;
    this.generatingScenario = true;
    this.generatedDescription = '';
    this.error = '';
    this.addLog('🎲 Generating new scenario via LLM...');
    this.cdr.detectChanges();

    this.coach.generateScenario(this.selectedDifficulty).subscribe({
      next: (res) => {
        this.zone.run(() => {
          this.sessionId = res.session_id;
          this.scenario = res.scenario;
          this.generatedDescription = res.short_description || '';
          this._setStartingMood(res.starting_mood); this.turnsRemaining = 20;
          this.generatingScenario = false;
          this.messages = [];
          this.resolved = false;
          this.failed = false;
          this.report = null;
          this.showReport = false;
          this.addLog(`🎲 Generated: ${res.scenario?.customer_persona} | ${res.scenario?.issue_type}`);
          this.cdr.detectChanges();
          this._connectWebSocket(res.session_id);
        });
      },
      error: (err) => {
        this.zone.run(() => {
          this.generatingScenario = false;
          this.addLog(`❌ Scenario generation failed — ${err.status}`);
          this.error = 'Failed to generate scenario. Is LM Studio running?';
          this.cdr.detectChanges();
        });
      }
    });
  }

  // ════════════════════════════════════════════════════════════════════════════
  //  Feature 1b: Start a user-defined custom scenario
  // ════════════════════════════════════════════════════════════════════════════
  toggleCustomForm() {
    this.showCustomForm = !this.showCustomForm;
    this.cdr.detectChanges();
  }

  startCustomScenario() {
    if (!this.customIssue.trim() || this.loading) return;
    this.loading = true;
    this.error = '';
    this.generatedDescription = '';
    this.addLog(`📝 Creating custom scenario: ${this.customIssue.trim()}`);
    this.cdr.detectChanges();

    this.coach.customScenario(
      this.customIssue.trim(),
      this.customPersona.trim(),
      this.customDescription.trim(),
      this.selectedDifficulty
    ).subscribe({
      next: (res) => {
        this.zone.run(() => {
          this.sessionId = res.session_id;
          this.scenario = res.scenario;
          this.generatedDescription = res.short_description || '';
          this._setStartingMood(res.starting_mood); this.turnsRemaining = 20;
          this.loading = false;
          this.messages = [];
          this.resolved = false;
          this.failed = false;
          this.report = null;
          this.showReport = false;
          this.showCustomForm = false;
          this.addLog(`✅ Custom: ${res.scenario?.customer_persona} | ${res.scenario?.issue_type}`);
          this.cdr.detectChanges();
          this._connectWebSocket(res.session_id);
        });
      },
      error: (err) => {
        this.zone.run(() => {
          this.loading = false;
          this.addLog(`❌ Custom scenario failed — ${err.status}`);
          this.error = `Custom scenario failed (${err.status})`;
          this.cdr.detectChanges();
        });
      }
    });
  }

  // ════════════════════════════════════════════════════════════════════════════
  //  Feature 2: Edit a previously sent agent reply
  // ════════════════════════════════════════════════════════════════════════════
  startEdit(index: number) {
    if (this.loading || this.resolved || this.failed) return;
    this.messages.forEach(m => { m.editing = false; m.editText = undefined; });
    this.messages[index].editing = true;
    this.messages[index].editText = this.messages[index].text;
    this.cdr.detectChanges();
  }

  cancelEdit(index: number) {
    this.messages[index].editing = false;
    this.messages[index].editText = undefined;
    this.cdr.detectChanges();
  }

  submitEdit(index: number) {
    const msg = this.messages[index];
    const newText = msg.editText?.trim();
    if (!newText || this.loading) return;
    if (newText === msg.text) {
      this.cancelEdit(index);
      return;
    }

    const turnNumber = Math.floor(index / 2) + 1;

    this.loading = true;
    this.processingLabel = `Re-evaluating turn ${turnNumber}...`;
    this.messages[index].text = newText;
    this.messages[index].editing = false;
    this.messages[index].editText = undefined;
    this.addLog(`✏️ Editing turn ${turnNumber}...`);
    this.cdr.detectChanges();

    this.coach.editMessage(this.sessionId, turnNumber, newText).subscribe({
      next: () => {
        this.processingLabel = 'Regenerating response...';
        this.cdr.detectChanges();
      },
      error: (err) => {
        this.addLog(`❌ Edit failed — ${err.status}`);
        this.error = `Edit failed (${err.status})`;
        this.loading = false;
        this.processingLabel = '';
        this.cdr.detectChanges();
      }
    });
  }

  // ════════════════════════════════════════════════════════════════════════════
  //  Feature 3: Redo conversation with the same scenario
  // ════════════════════════════════════════════════════════════════════════════
  redoConversation() {
    if (!this.sessionId || this.loading) return;
    this.loading = true;
    this.error = '';
    this.processingLabel = 'Restarting conversation...';
    this.addLog('🔄 Restarting conversation (same scenario)...');
    this.cdr.detectChanges();

    this.coach.redoConversation(this.sessionId).subscribe({
      next: (res) => {
        this.zone.run(() => {
          this.messages = [];
          this.resolved = false;
          this.failed = false;
          this.report = null;
          this.reportLoading = false;
          this.showReport = false;
          this.currentMood = res.starting_mood ?? 5;
          this.currentMoodLabel = this._moodLabelFor(this.currentMood);
          this.turnsRemaining = 20;
          this.loading = false;
          this.processingLabel = '';
          this.generatedDescription = '';
          this.addLog('✅ Conversation restarted — same scenario, fresh start');
          this.cdr.detectChanges();
        });
      },
      error: (err) => {
        this.zone.run(() => {
          this.loading = false;
          this.processingLabel = '';
          this.addLog(`❌ Redo failed — ${err.status}`);
          this.error = `Redo failed (${err.status})`;
          this.cdr.detectChanges();
        });
      }
    });
  }

  // ════════════════════════════════════════════════════════════════════════════
  //  NEW — Feature 4: Back to Chat — navigate from report to chat view
  // ════════════════════════════════════════════════════════════════════════════
  backToChat() {
    this.showReport = false;
    this.addLog('💬 Returned to chat view');
    this.cdr.detectChanges();
  }
}