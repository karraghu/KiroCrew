/**
 * Roster fixture shared by the Crew Members capture scripts
 * (capture-members-page.mjs, capture-members-steer-only.mjs): four members,
 * radar working. One copy, so the frames of different PRs photograph the
 * same crew and the copy-paste gate has nothing to find.
 */
export const MEMBERS = [
  { name: 'radar', slug: 'radar', bound: true, slot_key: 'member-radar', running: true, kiro_agent: 'kirocrew-autofix', workspace: 'autofix', memory_store: 'default', model: '', last_active_ts: 1000, last_message: 'Six new issues: four covered by open PRs.' },
  { name: 'scout', slug: 'scout', bound: false, slot_key: '', running: false, kiro_agent: 'kirocrew-research', workspace: 'default', memory_store: 'scout-own', model: 'claude-opus-5' },
  { name: 'fixer', slug: 'fixer', bound: true, slot_key: 'member-fixer', running: false, kiro_agent: 'kirocrew', workspace: 'default', memory_store: 'default', model: '', last_active_ts: 900, last_message: 'Two PRs opened for the queue.' },
  { name: 'scribe', slug: 'scribe', bound: false, slot_key: '', running: false, kiro_agent: 'kirocrew-lite', workspace: 'docs', memory_store: 'default', model: '' },
]
