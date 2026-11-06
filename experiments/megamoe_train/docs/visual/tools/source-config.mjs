// Copyright (c) 2026, Lu Lu
// Modified by xixing 2026

export function sourceConfig(environment, { remote, branch, commit }) {
  const repository = (environment.SOURCE_REPOSITORY_URL
    || (environment.GITHUB_REPOSITORY && `https://github.com/${environment.GITHUB_REPOSITORY}`)
    || remote.replace(/^git@([^:]+):/, 'https://$1/').replace(/^ssh:\/\/git@([^/]+)\//, 'https://$1/'))
    .replace(/\/$/, '').replace(/\.git$/, '');
  if (!/^https:\/\/[^/]+\/[^/]+\/[^/]+$/.test(repository)) {
    throw new Error('Expected an HTTPS owner/repository source URL');
  }
  const sourceBranch = environment.SOURCE_BRANCH || environment.GITHUB_HEAD_REF
    || environment.GITHUB_REF_NAME || branch || commit;
  // Online navigation is opt-in. Repository ownership does not enable a site.
  const configuredSite = (environment.SITE_URL || '').trim();
  let siteUrl = '';
  if (configuredSite) {
    const parsed = new URL(configuredSite);
    if (!['https:', 'http:'].includes(parsed.protocol) || parsed.search || parsed.hash) {
      throw new Error('Expected an HTTP(S) site URL without query or fragment');
    }
    siteUrl = configuredSite.replace(/\/*$/, '/');
  }
  return { repository, sourceBranch, siteUrl };
}
