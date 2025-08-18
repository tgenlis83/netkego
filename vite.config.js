import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwind from '@tailwindcss/vite'
import { fileURLToPath, URL } from 'node:url'
import fs from 'node:fs'
import path from 'node:path'
import { spawn } from 'node:child_process'

// https://vite.dev/config/
function apiPlugin() {
  return {
    name: 'networkego-api-plugin',
    configureServer(server) {
      const out = (p) => path.resolve(process.cwd(), p)
  const LOG_PATH = out('.runner/run_output.txt')
  const ensureLogDir = () => fs.mkdirSync(path.dirname(LOG_PATH), { recursive: true })
      const ensureRunnerPkg = () => {
        const pkgHidden = out('.runner/__init__.py')
        fs.mkdirSync(path.dirname(pkgHidden), { recursive: true })
        if (!fs.existsSync(pkgHidden)) fs.writeFileSync(pkgHidden, '', 'utf8')
        const pkgVisible = out('runner/__init__.py')
        fs.mkdirSync(path.dirname(pkgVisible), { recursive: true })
        if (!fs.existsSync(pkgVisible)) fs.writeFileSync(pkgVisible, '', 'utf8')
      }
  // Save generated code into .runner/generated_block.py
      server.middlewares.use('/api/save-generated', (req, res, next) => {
        if (req.method !== 'PUT') return next()
        let body = ''
        req.setEncoding('utf8')
        req.on('data', (c) => (body += c))
        req.on('end', () => {
          try {
            const outHidden = out('.runner/generated_block.py')
            const outVisible = out('runner/generated_block.py')
            fs.mkdirSync(path.dirname(outHidden), { recursive: true })
            fs.mkdirSync(path.dirname(outVisible), { recursive: true })
            ensureRunnerPkg()
            fs.writeFileSync(outHidden, body, 'utf8')
            fs.writeFileSync(outVisible, body, 'utf8')
            res.statusCode = 200
            res.end('ok')
          } catch (e) {
            res.statusCode = 500
            res.end(String(e))
          }
        })
      })

      // Save main script into .runner/main.py
      server.middlewares.use('/api/save-main', (req, res, next) => {
        if (req.method !== 'PUT') return next()
        let body = ''
        req.setEncoding('utf8')
        req.on('data', (c) => (body += c))
        req.on('end', () => {
          try {
            const outPath = out('.runner/main.py')
            fs.mkdirSync(path.dirname(outPath), { recursive: true })
            ensureRunnerPkg()
            fs.writeFileSync(outPath, body, 'utf8')
            res.statusCode = 200
            res.end('ok')
          } catch (e) {
            res.statusCode = 500
            res.end(String(e))
          }
        })
      })

    // Run python pipeline using uv (CPU) and pipe colored output to a hidden log file (avoid HMR reloads)
      server.middlewares.use('/api/run-python', (req, res, next) => {
        if (req.method !== 'POST') return next()
        try {
      ensureLogDir()
      fs.writeFileSync(LOG_PATH, '[server] Starting run...\n')

          const shell = process.env.SHELL || 'bash'
          const cmd = [
            '-lc',
            [
              'command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh',
              'export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"',
              'uv venv --seed .venv',
              '. .venv/bin/activate',
              'uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch torchvision tqdm',
              'python runner/run.py',
            ].join(' && '),
          ]
          const proc = spawn(shell, cmd, { stdio: ['ignore', 'pipe', 'pipe'] })
          const write = (chunk) => {
            ensureLogDir()
            fs.appendFileSync(LOG_PATH, chunk)
          }
          proc.stdout.on('data', (d) => write(d))
          proc.stderr.on('data', (d) => write(d))
          proc.on('close', (code) => {
            ensureLogDir()
            fs.appendFileSync(LOG_PATH, `\n[server] Process exited with code ${code}\n`)
          })
          res.statusCode = 200
          res.end('started')
        } catch (e) {
          res.statusCode = 500
          res.end(String(e))
        }
      })

      // Read the latest run output (no-store to avoid caching)
      server.middlewares.use('/api/run-output', (req, res, next) => {
        if (req.method !== 'GET') return next()
        try {
          ensureLogDir()
          const data = fs.existsSync(LOG_PATH) ? fs.readFileSync(LOG_PATH, 'utf8') : 'No runs yet.\n'
          res.statusCode = 200
          res.setHeader('Content-Type', 'text/plain; charset=utf-8')
          res.setHeader('Cache-Control', 'no-store')
          res.end(data)
        } catch (e) {
          res.statusCode = 500
          res.end(String(e))
        }
      })
    },
  }
}

export default defineConfig({
  plugins: [react(), tailwind(), apiPlugin()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    middlewareMode: false,
    watch: {
      ignored: [
        '**/.venv/**',
        '**/.runner/**',
        '**/runner/**',
      ],
    },
  },
})
