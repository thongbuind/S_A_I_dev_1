(function initSaiTheme() {
  const STORAGE_KEY = "sai-theme"
  const DEFAULT_THEME = "dark"
  const VALID_THEME_NAME = /^[a-z][a-z0-9-]*$/
  const root = document.documentElement

  function normalizeTheme(theme) {
    return typeof theme === "string" && VALID_THEME_NAME.test(theme)
      ? theme
      : DEFAULT_THEME
  }

  function readStoredTheme() {
    try {
      return normalizeTheme(localStorage.getItem(STORAGE_KEY))
    } catch {
      return DEFAULT_THEME
    }
  }

  function applyTheme(theme, { persist = false, notify = true } = {}) {
    const nextTheme = normalizeTheme(theme)
    const previousTheme = root.getAttribute("data-theme")

    root.setAttribute("data-theme", nextTheme)

    if (persist) {
      try {
        localStorage.setItem(STORAGE_KEY, nextTheme)
      } catch {}
    }

    if (notify && previousTheme !== nextTheme) {
      window.dispatchEvent(new CustomEvent("sai-theme-change", {
        detail: { theme: nextTheme },
      }))
    }

    return nextTheme
  }

  window.SAITheme = Object.freeze({
    get() {
      return normalizeTheme(root.getAttribute("data-theme"))
    },
    set(theme) {
      return applyTheme(theme, { persist: true })
    },
    toggle() {
      return applyTheme(this.get() === "dark" ? "light" : "dark", { persist: true })
    },
  })

  applyTheme(readStoredTheme(), { notify: false })

  window.addEventListener("storage", event => {
    if (event.key === STORAGE_KEY) applyTheme(event.newValue)
  })
})()
