// ============================================================
// FILE PATH: src/app/app.config.ts
// REPLACE the entire contents of this file with this code
// ============================================================
import { ApplicationConfig } from '@angular/core';
import { provideHttpClient, withFetch } from '@angular/common/http';

export const appConfig: ApplicationConfig = {
  providers: [
    provideHttpClient(withFetch())
  ]
};