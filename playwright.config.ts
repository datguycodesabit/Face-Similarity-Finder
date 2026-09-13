import { defineConfig, devices } from '@playwright/test';

const fixtureRoot = `${process.cwd()}/.browser-fixture`;

export default defineConfig({
  testDir: './tests/browser',
  outputDir: './artifacts/screenshots',
  fullyParallel: false,
  reporter: 'line',
  use: { baseURL: 'http://127.0.0.1:8765', trace: 'retain-on-failure' },
  webServer: {
    command: 'UV_CACHE_DIR=.uv-cache UV_PYTHON_INSTALL_DIR=.uv-python uv run python -m tests.browser_server',
    url: 'http://127.0.0.1:8765/api/status',
    reuseExistingServer: true,
    env: { BROWSER_FIXTURE_ROOT: fixtureRoot }
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }]
});
