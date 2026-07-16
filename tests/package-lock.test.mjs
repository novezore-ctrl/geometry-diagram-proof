import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const packageJson = JSON.parse(await readFile(new URL("../package.json", import.meta.url), "utf8"));
const packageLock = JSON.parse(await readFile(new URL("../package-lock.json", import.meta.url), "utf8"));
const root = packageLock.packages[""];

test("package metadata and lock root stay synchronized", () => {
  assert.equal(packageLock.name, packageJson.name);
  assert.equal(root.name, packageJson.name);
  assert.equal(packageLock.version, packageJson.version);
  assert.equal(root.version, packageJson.version);
  assert.deepEqual(root.dependencies || {}, packageJson.dependencies || {});
  assert.deepEqual(root.devDependencies || {}, packageJson.devDependencies || {});
});

test("removed starter dependencies do not remain in the lock", () => {
  assert.doesNotMatch(packageJson.name, /starter/i);
  assert.equal(packageJson.scripts["db:generate"], undefined);
  assert.equal(packageJson.dependencies["drizzle-orm"], undefined);
  assert.equal(packageJson.devDependencies["drizzle-kit"], undefined);
  assert.equal(Object.keys(packageLock.packages).some((path) => /(?:^|\/)drizzle(?:-|\/)/.test(path)), false);
});
