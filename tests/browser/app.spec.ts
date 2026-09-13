import { expect, test } from '@playwright/test';
import path from 'node:path';

const fixtureRoot = path.join(process.cwd(), '.browser-fixture');

const views = {
  frontInput: path.join(fixtureRoot, 'query_front.png'),
  leftInput: path.join(fixtureRoot, 'query_left.png'),
  rightInput: path.join(fixtureRoot, 'query_right.png')
};

async function selectThreeViews(page) {
  for (const [id, fixture] of Object.entries(views)) await page.locator(`#${id}`).setInputFiles(fixture);
}

test('three-view analysis shows shape, haircut guidance, matches, and overlays', async ({ page }, testInfo) => {
  await page.goto('/');
  await expect(page.getByText('6 eligible identities')).toBeVisible();
  await expect(page.getByRole('heading', { name: /Find faces shaped like yours/ })).toBeVisible();

  await selectThreeViews(page);
  await page.getByText('Optional haircut preferences').click();
  await page.locator('select[name="length"]').selectOption('medium');
  await page.locator('select[name="texture"]').selectOption('wavy');
  await page.locator('select[name="effort"]').selectOption('moderate');
  await page.locator('select[name="goal"]').selectOption('add-width');
  await expect(page.locator('.view-drop.has-image')).toHaveCount(3);
  await page.getByRole('button', { name: /Analyze face shape/ }).click();

  await expect(page.locator('.result-card')).toHaveCount(5);
  await expect(page.locator('.haircut-card')).toHaveCount(3);
  await expect(page.locator('#shapeBlend')).toContainText(/Mostly/);
  await expect(page.locator('#diagnosticsGrid > div')).toHaveCount(3);
  await expect(page.locator('#loadingBox')).toBeHidden();
  await expect(page.locator('.result-card canvas').first()).toBeVisible();
  await page.getByText('Dense mesh', { exact: true }).click();
  await expect(page.locator('.result-card canvas').first()).toBeHidden();
  await page.locator('#frontInput').setInputFiles(views.rightInput);
  await expect(page.locator('#resultsSection')).toBeHidden();
  await expect(page.locator('#selectionNote')).toContainText('Photos changed');
  await expect(page.getByRole('button', { name: /Analyze new photos/ })).toBeEnabled();
  await page.screenshot({ path: testInfo.outputPath('desktop-v2-results.png'), fullPage: true });
});

test('incorrect pose guidance can be corrected without reselecting photos', async ({ page }) => {
  await page.goto('/');
  await selectThreeViews(page);
  let rejectOnce = true;
  await page.route('**/api/analyze', async route => {
    if (rejectOnce) {
      rejectOnce = false;
      await route.fulfill({
        status: 422,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'The left photo should be a 15–35° three-quarter view.' })
      });
      return;
    }
    await route.continue();
  });
  await page.getByRole('button', { name: /Analyze face shape/ }).click();
  await expect(page.locator('#errorBox')).toContainText('15–35° three-quarter view');
  await page.getByRole('button', { name: /Analyze face shape/ }).click();
  await expect(page.locator('.result-card')).toHaveCount(5);
  await expect(page.locator('#errorBox')).toBeHidden();
});

test('mobile layout supports the full matching flow', async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Add three angles' })).toBeVisible();
  await expect(page.locator('.workspace')).toHaveCSS('grid-template-columns', '362px');
  await selectThreeViews(page);
  await page.getByRole('button', { name: /Analyze face shape/ }).click();
  await expect(page.locator('.result-card')).toHaveCount(5);
  await expect(page.locator('.haircut-card')).toHaveCount(3);
  await page.screenshot({ path: testInfo.outputPath('mobile-v2-results.png'), fullPage: true });
});

test('unsupported uploads show an actionable error state', async ({ page }, testInfo) => {
  await page.goto('/');
  const fixture = path.join(fixtureRoot, 'unsupported.gif');
  await page.locator('#frontInput').setInputFiles(fixture);
  await page.locator('#leftInput').setInputFiles(views.leftInput);
  await page.locator('#rightInput').setInputFiles(views.rightInput);
  await page.getByRole('button', { name: /Analyze face shape/ }).click();
  await expect(page.locator('#errorBox')).toContainText('Use a JPEG, PNG, or WebP image.');
  await page.screenshot({ path: testInfo.outputPath('upload-error.png'), fullPage: true });
});
