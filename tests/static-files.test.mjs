import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'
import { resolveStaticFile } from '../scripts/static-files.mjs'

test('PDF preview serves only files beneath its build root', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'rl-static-'))
  const root = path.join(tmp, 'dist')
  const sibling = path.join(tmp, 'dist-backup')
  fs.mkdirSync(root)
  fs.mkdirSync(sibling)
  fs.writeFileSync(path.join(root, 'index.html'), 'home')
  fs.writeFileSync(path.join(root, 'chapter.html'), 'chapter')
  fs.writeFileSync(path.join(sibling, 'fixture.txt'), 'test fixture')
  fs.symlinkSync(sibling, path.join(root, 'outside'), 'dir')
  try {
    assert.equal(resolveStaticFile(root, '/'), path.join(root, 'index.html'))
    assert.equal(
      resolveStaticFile(root, '/book/chapter', '/book/'),
      path.join(root, 'chapter.html')
    )
    assert.equal(
      resolveStaticFile(root, '/%2e%2e/dist-backup/fixture.txt'),
      null
    )
    assert.equal(resolveStaticFile(root, '/outside/fixture.txt'), null)
    assert.equal(resolveStaticFile(root, '/%ZZ'), null)
    assert.equal(resolveStaticFile(root, '/missing'), null)
    assert.equal(resolveStaticFile(root, '/%00'), null)
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true })
  }
})
