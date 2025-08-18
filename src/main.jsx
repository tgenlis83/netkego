import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'       // base resets (full-height root, neutral bg)
import './App.css'         // tailwind v4 (includes @import "tailwindcss")

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
