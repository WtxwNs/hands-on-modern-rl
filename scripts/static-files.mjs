import fs from 'node:fs'
import path from 'node:path'

function isWithin(root, filename) {
  const relative = path.relative(root, filename)
  return (
    relative !== '..' &&
    !relative.startsWith(`..${path.sep}`) &&
    !path.isAbsolute(relative)
  )
}

export function resolveStaticFile(distDir, urlPath, base = '/') {
  try {
    let pathname = decodeURIComponent(urlPath)
    if (base !== '/' && pathname.startsWith(base)) {
      pathname = `/${pathname.slice(base.length)}`
    }

    const root = fs.realpathSync(distDir)
    let filePath = path.resolve(
      root,
      `.${pathname.startsWith('/') ? '' : '/'}${pathname}`
    )
    if (!isWithin(root, filePath)) return null

    if (fs.existsSync(filePath) && fs.statSync(filePath).isDirectory()) {
      filePath = path.join(filePath, 'index.html')
    } else if (!fs.existsSync(filePath) && !path.extname(filePath)) {
      const cleanUrlPath = `${filePath}.html`
      filePath = fs.existsSync(cleanUrlPath)
        ? cleanUrlPath
        : path.join(filePath, 'index.html')
    }

    const resolved = fs.realpathSync(filePath)
    return isWithin(root, resolved) && fs.statSync(resolved).isFile()
      ? resolved
      : null
  } catch {
    return null
  }
}
