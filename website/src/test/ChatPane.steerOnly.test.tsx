import { describe, it, expect, vi, beforeEach } from 'vitest'
import type { ReactNode } from 'react'
import { act, render, screen, fireEvent, waitFor } from '@testing-library/react'
import type { RootState } from '../store'
import { Provider } from 'react-redux'
import { MemoryRouter } from 'react-router-dom'
import { configureStore } from '@reduxjs/toolkit'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ThemeProvider } from '../hooks/useTheme'
import chatReducer, { appendSlotMessage, selectSlotMessages, setActiveSlot, sseChatMessage } from '../store/chatSlice'
import type { SendReceipt, SendTurnOptions } from '../chat-core/transport/sendTurn'
import dashboardReducer from '../store/dashboardSlice'
import notificationsReducer from '../store/notificationsSlice'

/* Crew Members DM threads have no queue concept: a DM is a conversation with
 * ONE named member, and talking to a person has no "wait until they finish"
 * step. The Members page mounts ChatPane with busyMode="steer-only", so while
 * the member is working the composer keeps the plain send button and every
 * send is a STEER into the running turn — no Steer/Queue split, no QueueStack.
 *
 * The main chat and split view (⌘D) are unchanged: a ChatPane without the prop
 * keeps the split button, pinned here as the regression guard.
 *
 * Mutation checks: drop `busyMode={busyMode}` from the pane's ChatInput ->
 * tests 1-2 RED (split button appears); drop `steer: true` from doSteer's
 * sendTurn call -> test 1 RED (6th sendChat arg); drop `canSteer`/`onSteer`
 * from the pane's ChatInput -> test 3 RED (queue-only button, no caret). */

vi.mock('react-virtuoso', () => ({
  Virtuoso: ({ data, itemContent }: { data?: unknown[]; itemContent: (index: number, item: unknown) => ReactNode }) => (
    <div data-testid="virtuoso">{data?.map((d: unknown, i: number) => <div key={i}>{itemContent(i, d)}</div>)}</div>
  ),
}))
vi.mock('../api/client', () => ({
  api: {
    chatSlots: vi.fn().mockResolvedValue([]),
    chatSlotDetail: vi.fn().mockResolvedValue({ messages: [], running: true, has_more: false, total: 0 }),
    sendChat: vi.fn().mockResolvedValue({ ok: true, json: () => Promise.resolve({ ok: true, steered: true }) }),
    chatHistory: vi.fn().mockResolvedValue({ sessions: [] }),
    models: vi.fn().mockResolvedValue([]),
    agents: vi.fn().mockResolvedValue([]),
    agentDetail: vi.fn().mockResolvedValue({}),
    workspaces: vi.fn().mockResolvedValue({ workspaces: [] }),
    spawnList: vi.fn().mockResolvedValue({ agents: [] }),
    uploadFiles: vi.fn().mockResolvedValue({ paths: [] }),
    screenshot: vi.fn().mockResolvedValue({ path: null }),
    fileSearch: vi.fn().mockResolvedValue({ root: '/repo', results: [] }),
    chatSlotAgent: vi.fn().mockResolvedValue(undefined),
    editQueuedMessage: vi.fn().mockResolvedValue({ ok: true }),
    cancelQueuedMessage: vi.fn().mockResolvedValue({ ok: true }),
    interruptSlot: vi.fn().mockResolvedValue({ ok: true }),
    reorderQueuedMessages: vi.fn().mockResolvedValue({ ok: true }),
  },
  SEARCH_MIN_CHARS: 2,
  ApiError: class ApiError extends Error {
    status: number
    body: string
    constructor(status: number, message: string, body = '') {
      super(message)
      this.name = 'ApiError'
      this.status = status
      this.body = body
    }
  },
}))
vi.mock('../hooks/useVoiceInput', () => ({ useVoiceInput: () => ({ recording: false, transcribing: false, toggle: vi.fn() }), voiceInputSupported: false }))
vi.mock('../hooks/useBranding', () => ({ useBranding: () => ({ botName: 'Test', avatar: '' }) }))
vi.mock('../hooks/useAgents', () => ({ useAgents: () => ({ agents: [{ name: 'default' }], defaultAgent: 'default' }) }))
vi.mock('../components/MarkdownRenderer', () => ({ default: ({ content }: { content: string }) => <span>{content}</span> }))
vi.mock('../hooks/useWebSocket', () => ({ useWebSocket: () => ({ subscribeLogs: () => {} }) }))

/** Pass-through on the chat-core transport with ONE override hook: a test that
 *  needs an outcome the mocked fetch cannot produce deterministically (the
 *  abort deadline -> `response-late`) forces the receipt here instead of
 *  driving fake timers through react-query. Every other test hits the real
 *  `sendTurn` and asserts on `api.sendChat`'s arguments. */
let forcedReceipt: SendReceipt | null = null
let lastSendTurnOpts: SendTurnOptions | null = null
vi.mock('../chat-core/transport/sendTurn', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../chat-core/transport/sendTurn')>()
  return {
    ...actual,
    sendTurn: (opts: SendTurnOptions) => {
      lastSendTurnOpts = opts
      return forcedReceipt ? Promise.resolve(forcedReceipt) : actual.sendTurn(opts)
    },
  }
})

Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockReturnValue({ matches: false, addEventListener: vi.fn(), removeEventListener: vi.fn() }),
})

import ChatPane from '../components/ChatPane'
import { api } from '../api/client'

type BusyMode = 'split' | 'steer-only'

function makeStore(slotKey: string, running: boolean, subagentsOnly = false) {
  const store = configureStore({
    reducer: { dashboard: dashboardReducer, chat: chatReducer, notifications: notificationsReducer },
    preloadedState: {
      dashboard: {
        status: null, connected: true,
        // `subagents_running` is the slots-stream snapshot flag selectComposerBusy
        // reads: busy WITHOUT a running main turn (spawn_run is fire-and-forget).
        slots: [{ key: slotKey, messages: 0, running, subagents_running: subagentsOnly, mode: 'member', pending_approval: false, waiting_for_input: false, last_activity_ts: undefined }],
        unreadSlots: [], refreshTrigger: 0, approvalMode: 'normal',
        subagentRunning: {}, subagentDetails: {}, subagentText: {},
      } as unknown as RootState['dashboard'],
    } as Partial<RootState>,
  })
  // The pane is a BACKGROUND slot (the Members page never makes the DM the
  // active chat slot). A streamed chunk on it is what makes the pane's main
  // turn "running" in the store — the same frame the live WS delivers.
  store.dispatch(setActiveSlot('front'))
  if (running) store.dispatch(sseChatMessage({ slot: slotKey, role: 'chunk', content: 'working…', seq: 1 }))
  return store
}

function renderPane(slotKey: string, opts: { running: boolean; busyMode?: BusyMode; subagentsOnly?: boolean }) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const store = makeStore(slotKey, opts.running, opts.subagentsOnly)
  const ui = (key: string) => (
    <Provider store={store}>
      <QueryClientProvider client={qc}>
        <ThemeProvider>
          <MemoryRouter>
            <ChatPane slotKey={key} {...(opts.busyMode ? { busyMode: opts.busyMode } : {})} />
          </MemoryRouter>
        </ThemeProvider>
      </QueryClientProvider>
    </Provider>
  )
  const r = render(ui(slotKey))
  // Rebind the SAME pane instance to another slot — what the Members page does
  // when the user clicks another member (no `key`, so no remount).
  const rebind = (key: string) => r.rerender(ui(key))
  return Object.assign(r, { store, rebind })
}

const composer = async () => (await screen.findAllByRole('textbox'))[0]

beforeEach(() => {
  vi.clearAllMocks()
  localStorage.clear()
  sessionStorage.clear()
  forcedReceipt = null
  lastSendTurnOpts = null
  vi.mocked(api.sendChat).mockResolvedValue({ ok: true, json: () => Promise.resolve({ ok: true, steered: true }) } as unknown as Response)
  // clearAllMocks keeps implementations; a test that seeds slot-detail rows
  // must not leak them into the next one.
  vi.mocked(api.chatSlotDetail).mockResolvedValue({ messages: [], running: true, has_more: false, total: 0 } as never)
})

describe('ChatPane busyMode="steer-only" (Crew Members DM thread)', () => {
  it('busy member: plain send button, no split/queue affordance, and Enter steers into the running turn', async () => {
    const { store } = renderPane('member-oncall', { running: true, busyMode: 'steer-only' })
    const box = await composer()
    fireEvent.change(box, { target: { value: 'also check the logs' } })

    // The composer offers ONE control, the same send button an idle pane has.
    const send = screen.getByTestId('steer-only-send')
    expect(send).toHaveAttribute('aria-label', 'Send')
    expect(screen.queryByTestId('busy-send-button')).not.toBeInTheDocument()
    expect(screen.queryByTestId('busy-send-caret')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Queue message' })).not.toBeInTheDocument()

    fireEvent.keyDown(box, { key: 'Enter', code: 'Enter' })
    await waitFor(() => expect(api.sendChat).toHaveBeenCalledTimes(1))
    const [wireText, slot, , , meta, steer] = vi.mocked(api.sendChat).mock.calls[0]
    expect(wireText).toBe('also check the logs')
    expect(slot).toBe('member-oncall')
    expect(steer).toBe(true)
    expect(meta).toEqual({ sendId: expect.stringMatching(/^s-/) })

    // The steer shows up at once as an optimistic steer bubble on THIS slot…
    const rows = selectSlotMessages(store.getState() as RootState, 'member-oncall')
    const bubble = rows.find(m => m.role === 'user' && m.content === 'also check the logs')
    expect(bubble?.meta).toMatchObject({ steer: true, optimistic: true, sendId: (meta as { sendId: string }).sendId })
    // …and nothing was queued: no queue card anywhere in the pane.
    expect(screen.queryByRole('button', { name: 'Cancel queued message' })).not.toBeInTheDocument()
    expect((box as HTMLTextAreaElement).value).toBe('')
  })

  it('idle member: a plain send, no steer flag (steer-only changes only the BUSY composer)', async () => {
    renderPane('member-idle', { running: false, busyMode: 'steer-only' })
    const box = await composer()
    fireEvent.change(box, { target: { value: 'hello' } })
    fireEvent.keyDown(box, { key: 'Enter', code: 'Enter' })
    await waitFor(() => expect(api.sendChat).toHaveBeenCalledTimes(1))
    const [, , , , , steer] = vi.mocked(api.sendChat).mock.calls[0]
    expect(steer).toBeUndefined()
    expect(screen.queryByTestId('steer-only-send')).not.toBeInTheDocument()
  })

  it('default busyMode (split view, ⌘D) keeps the Steer/Queue split button on a busy pane', async () => {
    renderPane('pane-split', { running: true })
    const box = await composer()
    fireEvent.change(box, { target: { value: 'more' } })
    expect(screen.getByTestId('busy-send-button')).toHaveAttribute('aria-label', 'Steer')
    expect(screen.getByTestId('busy-send-caret')).toBeInTheDocument()
    expect(screen.queryByTestId('steer-only-send')).not.toBeInTheDocument()
    // The split's default action is the same steer path the DM uses.
    fireEvent.click(screen.getByTestId('busy-send-button'))
    await waitFor(() => expect(api.sendChat).toHaveBeenCalledTimes(1))
    expect(vi.mocked(api.sendChat).mock.calls[0][5]).toBe(true)
  })

  it('a refused steer drops the bubble, reports on this pane, and hands the text back', async () => {
    vi.mocked(api.sendChat).mockResolvedValue({ ok: false, status: 409, json: () => Promise.resolve({ error: 'slot agent mismatch' }) } as unknown as Response)
    const { store } = renderPane('member-refused', { running: true, busyMode: 'steer-only' })
    const box = await composer()
    fireEvent.change(box, { target: { value: 'try again' } })
    fireEvent.keyDown(box, { key: 'Enter', code: 'Enter' })
    await waitFor(() => expect(api.sendChat).toHaveBeenCalledTimes(1))
    await waitFor(() => expect((box as HTMLTextAreaElement).value).toBe('try again'))
    const rows = selectSlotMessages(store.getState() as RootState, 'member-refused')
    expect(rows.find(m => m.role === 'user' && m.meta?.steer)).toBeUndefined()
    expect(rows.find(m => m.role === 'error')?.content).toContain('slot agent mismatch')
  })

  it('a confirmed steer renders as an ordinary message — no "Steered into the running turn" badge', async () => {
    const { store } = renderPane('member-badge', { running: true, busyMode: 'steer-only' })
    await composer()
    // The server's steer_push echo, as the WS delivers it for this slot.
    act(() => {
      store.dispatch(appendSlotMessage({
        slot: 'member-badge',
        message: { role: 'user', content: 'also check the logs', cls: 'msg msg-u', ts: '2026-09-06T00:00:10Z', meta: { steer: true, steerState: 'consumed', sendId: 's-echo' } },
      }))
    })
    expect(await screen.findByText('also check the logs')).toBeInTheDocument()
    expect(screen.queryByText('Steered into the running turn')).not.toBeInTheDocument()
  })

  it('default busyMode keeps the steer badge on a confirmed steer (split view unchanged)', async () => {
    const { store } = renderPane('pane-badge', { running: true })
    await composer()
    act(() => {
      store.dispatch(appendSlotMessage({
        slot: 'pane-badge',
        message: { role: 'user', content: 'also check the logs', cls: 'msg msg-u', ts: '2026-09-06T00:00:10Z', meta: { steer: true, steerState: 'consumed', sendId: 's-echo' } },
      }))
    })
    expect(await screen.findByText('also check the logs')).toBeInTheDocument()
    expect(screen.getByText('Steered into the running turn')).toBeInTheDocument()
  })

  it('sub-agents-only busy: a send whose receipt never came hands the draft back and warns instead of losing it', async () => {
    // Busy WITHOUT a running main turn: only sub-agents are active, so the
    // steer takes the send-with-steer-flag path, which mints no optimistic
    // bubble. The transport's deadline then fires (response-late) — forced
    // through the transport hook, since nothing on screen represents the text.
    forcedReceipt = { status: 'response-late', body: {} }
    const { store } = renderPane('member-subagents', { running: false, busyMode: 'steer-only', subagentsOnly: true })
    const box = await composer()
    fireEvent.change(box, { target: { value: 'while you wait, look at #77' } })
    fireEvent.keyDown(box, { key: 'Enter', code: 'Enter' })
    await waitFor(() => expect(lastSendTurnOpts).not.toBeNull())
    expect(lastSendTurnOpts?.steer).toBe(true)
    await waitFor(() => expect((box as HTMLTextAreaElement).value).toBe('while you wait, look at #77'))
    const rows = selectSlotMessages(store.getState() as RootState, 'member-subagents')
    expect(rows.some(m => m.role === 'notice')).toBe(true)
    expect(rows.some(m => m.role === 'user')).toBe(false)
  })

  it('an identical queue entry does not stand in for the unconfirmed send — the draft still comes back', async () => {
    // "ok" is already parked on the slot's queue. The user sends "ok" again;
    // the receipt never arrives. Queue cards carry no sendId, so NO card may be
    // read as this send's echo — only an id-bearing user echo can — or the
    // new draft is lost behind a card that may belong to anyone.
    vi.mocked(api.chatSlotDetail).mockResolvedValue({
      messages: [{ role: 'queued', content: 'ok', cls: 'msg msg-queued', ts: '2026-09-06T00:00:01Z', meta: { queueId: 'q-old' } }],
      running: false, has_more: false, total: 1,
    } as never)
    forcedReceipt = { status: 'response-late', body: {} }
    const { store } = renderPane('member-dup-queue', { running: false, busyMode: 'steer-only', subagentsOnly: true })
    const box = await composer()
    await screen.findByRole('button', { name: 'Cancel queued message' })
    fireEvent.change(box, { target: { value: 'ok' } })
    fireEvent.keyDown(box, { key: 'Enter', code: 'Enter' })
    await waitFor(() => expect(lastSendTurnOpts).not.toBeNull())
    await waitFor(() => expect((box as HTMLTextAreaElement).value).toBe('ok'))
    const rows = selectSlotMessages(store.getState() as RootState, 'member-dup-queue')
    expect(rows.some(m => m.role === 'notice')).toBe(true)
    // The pre-existing card is untouched.
    expect(rows.filter(m => m.role === 'queued')).toHaveLength(1)
  })

  it('a refused send frames the server reason with what happened and where the text went', async () => {
    vi.mocked(api.sendChat).mockResolvedValue({ ok: false, status: 409, json: () => Promise.resolve({ error: 'slot agent mismatch' }) } as unknown as Response)
    const { store } = renderPane('member-framed', { running: true, busyMode: 'steer-only' })
    const box = await composer()
    fireEvent.change(box, { target: { value: 'try again' } })
    fireEvent.keyDown(box, { key: 'Enter', code: 'Enter' })
    await waitFor(() => expect((box as HTMLTextAreaElement).value).toBe('try again'))
    const err = selectSlotMessages(store.getState() as RootState, 'member-framed').find(m => m.role === 'error')
    expect(err?.content).toBe("Couldn't send this message: slot agent mismatch. Your text is back in the composer.")
  })

  it('rebinding the pane to another member parks the draft — it does not ride into the other composer', async () => {
    const { rebind } = renderPane('member-a', { running: false, busyMode: 'steer-only' })
    const box = await composer()
    fireEvent.change(box, { target: { value: 'half-typed note for A' } })
    rebind('member-b')
    await waitFor(() => expect((box as HTMLTextAreaElement).value).toBe(''))
    fireEvent.change(box, { target: { value: 'something for B' } })
    rebind('member-a')
    await waitFor(() => expect((box as HTMLTextAreaElement).value).toBe('half-typed note for A'))
    rebind('member-b')
    await waitFor(() => expect((box as HTMLTextAreaElement).value).toBe('something for B'))
  })

  it('a refusal that lands after the user switched members restores into the SENDING member, not the one on screen', async () => {
    let refuse!: () => void
    vi.mocked(api.sendChat).mockReturnValue(new Promise((resolve) => {
      refuse = () => resolve({ ok: false, status: 409, json: () => Promise.resolve({ error: 'slot agent mismatch' }) } as unknown as Response)
    }) as ReturnType<typeof api.sendChat>)
    const { store, rebind } = renderPane('member-a', { running: true, busyMode: 'steer-only' })
    const box = await composer()
    fireEvent.change(box, { target: { value: 'meant for A' } })
    fireEvent.keyDown(box, { key: 'Enter', code: 'Enter' })
    await waitFor(() => expect(api.sendChat).toHaveBeenCalledTimes(1))
    // The user moves on to B while A's steer is still in flight...
    rebind('member-b')
    await waitFor(() => expect((box as HTMLTextAreaElement).value).toBe(''))
    // ...and A's send is refused now.
    await act(async () => { refuse() })
    await waitFor(() => expect(selectSlotMessages(store.getState() as RootState, 'member-a').some(m => m.role === 'error')).toBe(true))
    // B's composer is untouched; the error row went to A's transcript.
    expect((box as HTMLTextAreaElement).value).toBe('')
    expect(selectSlotMessages(store.getState() as RootState, 'member-b').some(m => m.role === 'error')).toBe(false)
    // Back on A, the text is waiting.
    rebind('member-a')
    await waitFor(() => expect((box as HTMLTextAreaElement).value).toBe('meant for A'))
  })

  it('an upload that finishes after the user switched members is staged for the member it was picked in', async () => {
    let finish!: () => void
    vi.mocked(api.uploadFiles).mockReturnValue(new Promise((resolve) => {
      finish = () => resolve({ paths: ['/tmp/uploads/report.pdf'] })
    }) as ReturnType<typeof api.uploadFiles>)
    const { container, rebind } = renderPane('member-a', { running: false, busyMode: 'steer-only' })
    await composer()
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement
    Object.defineProperty(fileInput, 'files', { value: [new File(['x'], 'report.pdf', { type: 'application/pdf' })] })
    fireEvent.change(fileInput)
    await waitFor(() => expect(api.uploadFiles).toHaveBeenCalledTimes(1))
    // Switch to B while A's upload is still in flight, then let it finish.
    rebind('member-b')
    await act(async () => { finish() })
    // B did not receive A's attachment…
    expect(screen.queryByText('report.pdf')).not.toBeInTheDocument()
    // …A has it waiting.
    rebind('member-a')
    expect(await screen.findByText('report.pdf')).toBeInTheDocument()
  })

  it('an upload that FAILS after the user switched members is reported when they come back, not over the other thread', async () => {
    let fail!: () => void
    vi.mocked(api.uploadFiles).mockReturnValue(new Promise((resolve) => {
      fail = () => resolve({ paths: [], error: 'Unsupported file type' })
    }) as ReturnType<typeof api.uploadFiles>)
    const { container, rebind } = renderPane('member-a', { running: false, busyMode: 'steer-only' })
    await composer()
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement
    Object.defineProperty(fileInput, 'files', { value: [new File(['x'], 'evil.exe')] })
    fireEvent.change(fileInput)
    await waitFor(() => expect(api.uploadFiles).toHaveBeenCalledTimes(1))
    rebind('member-b')
    await act(async () => { fail() })
    // Not over B…
    expect(screen.queryByTestId('pane-upload-error')).not.toBeInTheDocument()
    // …but waiting on A, through the shared error surface, dismissible.
    rebind('member-a')
    const notice = await screen.findByTestId('pane-upload-error')
    expect(notice).toHaveTextContent('Unsupported file type')
    fireEvent.click(screen.getByRole('button', { name: /dismiss/i }))
    await waitFor(() => expect(screen.queryByTestId('pane-upload-error')).not.toBeInTheDocument())
  })

  it('an upload that fails after the pane unmounted lands in that slot\'s transcript as an error row', async () => {
    let fail!: () => void
    vi.mocked(api.uploadFiles).mockReturnValue(new Promise((resolve) => {
      fail = () => resolve({ paths: [], error: 'Unsupported file type' })
    }) as ReturnType<typeof api.uploadFiles>)
    const first = renderPane('member-a', { running: false, busyMode: 'steer-only' })
    await composer()
    const fileInput = first.container.querySelector('input[type="file"]') as HTMLInputElement
    Object.defineProperty(fileInput, 'files', { value: [new File(['x'], 'evil.exe')] })
    fireEvent.change(fileInput)
    await waitFor(() => expect(api.uploadFiles).toHaveBeenCalledTimes(1))
    first.unmount()
    await act(async () => { fail() })
    const rows = selectSlotMessages(first.store.getState() as RootState, 'member-a')
    expect(rows.find(m => m.role === 'error')?.content).toContain('Unsupported file type')
  })

  it('a draft survives the pane unmounting (leaving the page) and a refusal that lands while it is gone', async () => {
    let refuse!: () => void
    vi.mocked(api.sendChat).mockReturnValue(new Promise((resolve) => {
      refuse = () => resolve({ ok: false, status: 409, json: () => Promise.resolve({ error: 'slot agent mismatch' }) } as unknown as Response)
    }) as ReturnType<typeof api.sendChat>)
    const first = renderPane('member-a', { running: true, busyMode: 'steer-only' })
    const box = await composer()
    fireEvent.change(box, { target: { value: 'sent, then I left' } })
    fireEvent.keyDown(box, { key: 'Enter', code: 'Enter' })
    await waitFor(() => expect(api.sendChat).toHaveBeenCalledTimes(1))
    fireEvent.change(box, { target: { value: 'still typing' } })
    // Leave the Members page: the pane unmounts with the send in flight…
    first.unmount()
    // …and the send is refused with nobody on screen to hand the text to.
    await act(async () => { refuse() })
    // Coming back, both the unsent typing and the refused text are waiting.
    renderPane('member-a', { running: false, busyMode: 'steer-only' })
    const again = await composer()
    await waitFor(() => expect((again as HTMLTextAreaElement).value).toContain('still typing'))
    expect((again as HTMLTextAreaElement).value).toContain('sent, then I left')
  })
})
