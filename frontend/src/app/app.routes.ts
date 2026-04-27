import { Routes } from '@angular/router';
import { authGuard } from './guards/auth.guard';

export const routes: Routes = [
  { path: '', redirectTo: 'login', pathMatch: 'full' },
  {
    path: 'login',
    loadComponent: () =>
      import('./components/login/login.component').then(m => m.LoginComponent),
  },
  {
    path: 'chat',
    canActivate: [authGuard],
    loadComponent: () =>
      import('./components/chat/chat.component').then(m => m.ChatComponent),
  },
  {
    path: 'history',
    canActivate: [authGuard],
    loadComponent: () =>
      import('./components/history/history.component').then(m => m.HistoryComponent),
  },
  {
    path: 'history/:sessionId',
    canActivate: [authGuard],
    loadComponent: () =>
      import('./components/session-report/session-report.component').then(
        m => m.SessionReportComponent,
      ),
  },
  { path: '**', redirectTo: 'login' },
];
