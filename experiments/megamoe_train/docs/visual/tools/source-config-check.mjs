// Copyright (c) 2026, Lu Lu
// Modified by xixing 2026

import assert from 'node:assert/strict';
import { sourceConfig } from './source-config.mjs';

const git = { remote: 'git@github.com:example/DeepEP.git', branch: 'main', commit: 'a'.repeat(40) };
const offline = sourceConfig({}, git);
assert.equal(offline.repository, 'https://github.com/example/DeepEP');
assert.equal(offline.sourceBranch, 'main');
assert.equal(offline.siteUrl, '');
assert.equal(sourceConfig({ SITE_URL: '' }, git).siteUrl, '', 'An empty site setting must remain offline');
assert.equal(sourceConfig({ GITHUB_REPOSITORY: 'upstream/DeepEP' }, git).siteUrl, '', 'CI must not infer a Pages URL');
assert.equal(sourceConfig({ SITE_URL: 'https://example.github.io/DeepEP' }, git).siteUrl, 'https://example.github.io/DeepEP/');
const configured = sourceConfig({ SOURCE_REPOSITORY_URL: 'https://github.com/upstream/DeepEP/',
  SOURCE_BRANCH: 'main', GITHUB_REPOSITORY: 'fork/DeepEP', GITHUB_HEAD_REF: 'feature' }, git);
assert.equal(configured.repository, 'https://github.com/upstream/DeepEP');
assert.equal(configured.sourceBranch, 'main');
assert.equal(sourceConfig({ GITHUB_HEAD_REF: 'feature' }, git).sourceBranch, 'feature');
assert.equal(sourceConfig({}, { ...git, branch: '' }).sourceBranch, git.commit, 'Detached source must not invent a branch');
assert.throws(() => sourceConfig({ SITE_URL: 'file:///tmp/guide' }, git), /HTTP/);
console.log('Source configuration PASS: offline default, explicit site and source identity.');
