/* Accounts — optional, via Supabase Auth.

   The server says whether sign-in is configured (/config). If it is not,
   none of this appears and supabase-js is never downloaded. If it is, the
   library signs the user in and keeps the session fresh; the server only
   ever sees the short-lived access token, as a Bearer header. */

const Account = (() => {
  const SDK =
    "https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2.112.0/dist/umd/supabase.min.js";
  // This library holds the session, so it is the last file that should be
  // swappable: the browser rejects it unless the bytes match this hash.
  const SDK_INTEGRITY =
    "sha384-De+l/Df7qym5QBDVocD3+gW1S7IjR3epf+6KbRDjB3FCCzM/LqVTwqckT6FhFar8";
  const el = (id) => document.getElementById(id);
  const listeners = [];
  let sb = null;
  let session = null;
  let mode = "sign-in";

  const MODES = {
    "sign-in": { title: "Sign in", submit: "Sign in", email: true, password: true,
                 toggle: "Create an account instead", forgot: true },
    "sign-up": { title: "Create an account", submit: "Create account", email: true,
                 password: true, toggle: "I already have an account" },
    "reset": { title: "Reset your password", submit: "Email me a link", email: true,
               toggle: "Back to sign in" },
    "new-password": { title: "Choose a new password", submit: "Save password",
                      password: true },
  };

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = src;
      s.integrity = SDK_INTEGRITY;
      s.crossOrigin = "anonymous";
      s.onload = resolve;
      s.onerror = () => reject(new Error("could not load the sign-in library"));
      document.head.appendChild(s);
    });
  }

  async function init() {
    const { auth } = await fetch("/config").then((r) => r.json());
    if (!auth) return;
    await loadScript(SDK);
    sb = window.supabase.createClient(auth.url, auth.publishable_key);
    wire();
    el("account").hidden = false;
    sb.auth.onAuthStateChange((event, s) => {
      session = s;
      if (event === "PASSWORD_RECOVERY") open("new-password");
      // Supabase advises against calling back into the client from inside
      // this callback, so listeners run on the next tick.
      setTimeout(() => {
        paint();
        listeners.forEach((fn) => fn(user()));
      }, 0);
    });
  }

  const user = () => (session ? session.user : null);
  const enabled = () => sb !== null;

  /* Fresh headers for an API call. getSession() refreshes an expired token,
     so a tab left open overnight still sends a valid one. */
  async function headers() {
    if (!sb) return {};
    const { data } = await sb.auth.getSession();
    session = data.session;
    return session ? { Authorization: `Bearer ${session.access_token}` } : {};
  }

  function onChange(fn) { listeners.push(fn); }

  /* ── Header ─────────────────────────────────────────────────────────── */

  function paint() {
    const u = user();
    el("who").textContent = u ? u.email : "";
    el("sign-in").hidden = !!u;
    el("sign-out").hidden = !u;
  }

  /* ── Dialog ─────────────────────────────────────────────────────────── */

  function open(next) {
    mode = next;
    const m = MODES[mode];
    el("auth-title").textContent = m.title;
    el("auth-submit").textContent = m.submit;
    el("auth-email-row").hidden = !m.email;
    el("auth-email").required = !!m.email;
    el("auth-password-row").hidden = !m.password;
    el("auth-password").required = !!m.password;
    el("auth-password").autocomplete =
      mode === "sign-in" ? "current-password" : "new-password";
    el("auth-toggle").hidden = !m.toggle;
    el("auth-toggle").textContent = m.toggle || "";
    el("auth-forgot").hidden = !m.forgot;
    say("");
    if (!el("auth-dialog").open) el("auth-dialog").showModal();
    (m.email ? el("auth-email") : el("auth-password")).focus();
  }

  function say(text, isError = false) {
    el("auth-status").textContent = text;
    el("auth-status").className = `status${isError ? " err" : ""}`;
  }

  async function submit(event) {
    event.preventDefault();
    const email = el("auth-email").value.trim();
    const password = el("auth-password").value;
    const btn = el("auth-submit");
    btn.disabled = true;
    try {
      if (mode === "sign-in") {
        const { error } = await sb.auth.signInWithPassword({ email, password });
        if (error) throw error;
        close();
      } else if (mode === "sign-up") {
        const { data, error } = await sb.auth.signUp({
          email, password, options: { emailRedirectTo: location.origin },
        });
        if (error) throw error;
        if (data.session) close();
        else say("Check your email for a confirmation link, then sign in.");
      } else if (mode === "reset") {
        const { error } = await sb.auth.resetPasswordForEmail(email, {
          redirectTo: location.origin,
        });
        if (error) throw error;
        // Same message either way: do not reveal which addresses have accounts.
        say("If that address has an account, a reset link is on its way.");
      } else if (mode === "new-password") {
        const { error } = await sb.auth.updateUser({ password });
        if (error) throw error;
        close();
      }
    } catch (err) {
      say(err.message || String(err), true);
    } finally {
      btn.disabled = false;
    }
  }

  function close() {
    el("auth-password").value = "";
    el("auth-dialog").close();
  }

  function wire() {
    el("sign-in").addEventListener("click", () => open("sign-in"));
    el("sign-out").addEventListener("click", () => sb.auth.signOut());
    el("auth-form").addEventListener("submit", submit);
    el("auth-cancel").addEventListener("click", close);
    el("auth-forgot").addEventListener("click", () => open("reset"));
    el("auth-toggle").addEventListener("click", () =>
      open(mode === "sign-in" ? "sign-up" : "sign-in"));
  }

  return { init, headers, onChange, user, enabled };
})();
