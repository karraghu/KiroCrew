/**
 * Per-slot composer drafts for ChatPane — the pane's instance of the repo's
 * slot-draft store (`createSlotDraftStore`), alongside `chatDrafts` (ChatPage
 * text) and `chatFileDrafts` (ChatPage attachments).
 *
 * Why a separate KEY rather than ChatPage's stores: ChatPage holds its draft
 * maps in memory and persists them wholesale on its own schedule, so a second
 * writer on the same key would be overwritten by ChatPage's next save (and
 * ChatPage would never see the pane's write until a reload). The pane instead
 * does read-modify-write against its own keys on every access, so several
 * panes — split view — can share them safely.
 *
 * What it holds: the composer of every slot a pane is NOT currently showing.
 * A pane can be rebound to another slot without remounting (the Members page
 * switches `slotKey` on one instance), and a send's recovery can land after
 * that switch; both park here. The on-screen slot's composer is the live
 * React state — the store is authoritative only for off-screen slots, which
 * is why the pane writes on rebind and unmount rather than per keystroke.
 *
 * Same storage tiers as ChatPage's pair, for the same reasons: text in
 * localStorage with the shared TTL/LRU caps (survives refresh and leaving
 * the page); attachment paths in sessionStorage (upload paths are ephemeral).
 */
import { createSlotDraftStore } from './slotDraftStore'
import { DRAFT_MAX_ENTRIES, DRAFT_MAX_STORE_BYTES, DRAFT_TTL_MS } from './draftConstants'
import { mergeRecoveredDraft } from './chatDrafts'

export const PANE_DRAFTS_KEY = 'mc-pane-drafts'
export const PANE_FILE_DRAFTS_KEY = 'mc-pane-file-drafts'

const textStore = createSlotDraftStore<string>({
  key: PANE_DRAFTS_KEY,
  storage: 'local',
  ttlMs: DRAFT_TTL_MS,
  maxEntries: DRAFT_MAX_ENTRIES,
  maxStoreBytes: DRAFT_MAX_STORE_BYTES,
  sanitize: (v) => (typeof v === 'string' && v ? v : null),
})

const fileStore = createSlotDraftStore<string[]>({
  key: PANE_FILE_DRAFTS_KEY,
  storage: 'session',
  sanitize: (v) => {
    if (!Array.isArray(v)) return null
    const arr = v.filter((x): x is string => typeof x === 'string')
    return arr.length ? arr.slice() : null
  },
})

export interface PaneDraft {
  text: string
  files: string[]
}

/** The parked composer for `slot`, or empty. */
export function readPaneDraft(slot: string): PaneDraft {
  return {
    text: textStore.load()[slot] ?? '',
    files: fileStore.load()[slot] ?? [],
  }
}

/** Park `slot`'s composer verbatim. Empty text / no files delete the entry. */
export function writePaneDraft(slot: string, draft: PaneDraft): void {
  const texts = textStore.load()
  textStore.set(texts, slot, draft.text)
  textStore.save(texts)
  const files = fileStore.load()
  fileStore.set(files, slot, draft.files)
  fileStore.save(files)
}

/** Merge a late recovery (or a late upload) into `slot`'s parked composer:
 *  text appends under the shared recovery rule, paths union. */
export function mergePaneDraft(slot: string, text: string, files: string[]): void {
  const cur = readPaneDraft(slot)
  writePaneDraft(slot, {
    text: text ? mergeRecoveredDraft(cur.text, text) : cur.text,
    files: [...cur.files, ...files.filter((f) => !cur.files.includes(f))],
  })
}
