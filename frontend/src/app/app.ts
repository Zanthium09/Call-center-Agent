// ============================================================
// FILE PATH: src/app/app.ts
// REPLACE the entire contents of this file with this code
// ============================================================
import { Component } from '@angular/core';
import { ChatComponent } from './components/chat/chat.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [ChatComponent],
  template: '<app-chat></app-chat>'
})
export class App { }