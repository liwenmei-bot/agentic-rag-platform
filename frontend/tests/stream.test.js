import { afterEach, test } from 'node:test'
import assert from 'node:assert/strict'
import { streamAgenticChat } from '../src/api/client.js'

const originalFetch = globalThis.fetch
afterEach(() => { globalThis.fetch = originalFetch })

function response(chunks) {
  const encoder = new TextEncoder()
  return { ok: true, body: new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk))
      controller.close()
    },
  }) }
}

test('dispatches live frames across network chunks and CRLF boundaries', async () => {
  globalThis.fetch = async () => response([
    'data: {"type":"stage","data":{"stage":"router"}}\r',
    '\n\r\ndata: {"type":"content","data":"你好"}\r\n\r\n',
    'data: {"type":"done","data":{"status":"ok"}}\n\n',
  ])
  const events = []
  await streamAgenticChat('session', '问题', (event) => events.push(event))
  assert.deepEqual(events.map((e) => e.type), ['stage', 'content', 'done'])
  assert.equal(events[1].data, '你好')
})

test('rejects interrupted streams and server error events', async () => {
  globalThis.fetch = async () => response(['data: {"type":"content","data":"partial"}\n\n'])
  await assert.rejects(streamAgenticChat('session', '问题', () => {}), /中断/)
  globalThis.fetch = async () => response(['data: {"type":"error","data":{"message":"失败"}}\n\n'])
  await assert.rejects(streamAgenticChat('session', '问题', () => {}), /失败/)
})
