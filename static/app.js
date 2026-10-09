/* ragpdf — Jev gate then SSE search. No framework. */
(() => {
  const $ = (id) => document.getElementById(id);
  const f = $("f"), q = $("q"), doc = $("doc"), go = $("go");
  const prog = $("prog"), out = $("out"), ans = $("ans"), cite = $("cite");
  const evd = $("evd"), evdsum = $("evdsum"), thumbs = $("thumbs");
  const err = $("err"), meta = $("meta"), dot = $("dot");
  const clar = document.createElement("div");
  clar.id = "clar"; clar.hidden = true;
  f.after(clar);
  let es = null;
  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");

  fetch("/api/health").then((r) => r.json()).then((h) => {
    dot.className = "status-dot " + (h.ok ? "ok" : "down");
    if (h.ok) meta.textContent = h.points + " pages";
  }).catch(() => { dot.className = "status-dot down"; });

  fetch("/api/docs").then((r) => r.json()).then((d) => {
    if (!d.ok) return;
    refreshDocs(d);
  }).catch(() => {});

  function refreshDocs(d) {
    const keep = doc.value;
    doc.innerHTML = "<option value=''>All (" + d.count + ")</option>" +
      d.docs.map((x) => "<option value='" + esc(x.doc_id) + "'>" + esc(x.doc_id) + " · " + x.pages + "p</option>").join("");
    if ([...doc.options].some((o) => o.value === keep)) doc.value = keep;
    const total = d.docs.reduce((a, x) => a + x.pages, 0);
    meta.textContent = d.count + " doc" + (d.count > 1 ? "s" : "") + " · " + total + " pages";
  }

  // Upload: POST the PDF, then poll the background ingest job.
  const file = $("file"), job = $("job");
  file.addEventListener("change", () => {
    const pdf = file.files[0];
    if (!pdf) return;
    file.value = "";
    job.hidden = false;
    job.textContent = "Uploading " + pdf.name + "…";
    const fd = new FormData();
    fd.append("file", pdf, pdf.name);
    fetch("/api/upload", { method: "POST", body: fd }).then((r) => r.json()).then((u) => {
      if (!u.ok) { job.textContent = "Upload failed: " + (u.error || "unknown"); return; }
      pollJob(u.job_id);
    }).catch(() => { job.textContent = "Upload failed: connection error"; });
  });

  function pollJob(id) {
    fetch("/api/jobs/" + id).then((r) => r.json()).then((j) => {
      if (!j.ok) { job.textContent = "Job lost"; return; }
      const pct = j.total ? " · " + Math.round(100 * j.done / Math.max(j.total, 1)) + "%" : "";
      job.textContent = j.file + " — " + j.status + pct + (j.detail ? " · " + j.detail : "");
      if (j.status === "done") {
        fetch("/api/docs").then((r) => r.json()).then((d) => { if (d.ok) refreshDocs(d); }).catch(() => {});
        setTimeout(() => { job.hidden = true; }, 8000);
      } else if (j.status === "error") {
        job.textContent += " (retry: re-upload the file)";
      } else {
        setTimeout(() => pollJob(id), 2000);
      }
    }).catch(() => setTimeout(() => pollJob(id), 3000));
  }

  thumbs.addEventListener("click", (e) => {
    const img = e.target.closest("img");
    if (!img) return;
    const lb = document.createElement("div");
    lb.className = "lb";
    lb.innerHTML = "<img>";
    lb.firstChild.src = img.src;
    lb.onclick = () => lb.remove();
    document.addEventListener("keydown", function k2(e2) {
      if (e2.key === "Escape") { lb.remove(); document.removeEventListener("keydown", k2); }
    });
    document.body.appendChild(lb);
  });

  clar.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-s]");
    if (!b) return;
    q.value = b.dataset.s;
    clar.hidden = true;
    search(q.value.trim());
  });

  f.onsubmit = (e) => {
    e.preventDefault();
    const query = q.value.trim();
    if (!query || es || go.disabled) return;
    clar.hidden = true;
    err.hidden = true;
    go.disabled = true; go.textContent = "…";
    prog.hidden = false; prog.textContent = "Checking question…";
    fetch("/api/classify", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: query, doc_id: doc.value || undefined }),
    }).then((r) => r.json()).then((c) => {
      go.disabled = false; go.textContent = "Ask";
      if (c.ok && !c.clear) { askClarify(c); return; }
      search(query); // clear, or classifier failed open
    }).catch(() => {
      go.disabled = false; go.textContent = "Ask";
      search(query); // gate unreachable: fail open, never block search
    });
  };

  function askClarify(c) {
    prog.hidden = true;
    out.hidden = true;
    clar.innerHTML = "<p class='why'>" + esc(c.reason || "That could mean a few things.") + " Which did you mean?</p>" +
      (c.suggestions || []).map((s) => "<button type='button' data-s='" + esc(s) + "'>" + esc(s) + "</button>").join("") +
      "<button type='button' class='anyway' id='anyway'>Search anyway</button>";
    clar.hidden = false;
    clar.querySelector("#anyway").onclick = () => { clar.hidden = true; search(q.value.trim()); };
  }

  function search(query) {
    if (es) es.close();
    err.hidden = true; out.hidden = false; evd.open = false;
    ans.textContent = ""; cite.textContent = ""; thumbs.innerHTML = "";
    prog.hidden = false;
    go.disabled = true; go.textContent = "…";
    let buf = "";
    let url = "/api/ask/stream?q=" + encodeURIComponent(query);
    if (doc.value) url += "&doc_id=" + encodeURIComponent(doc.value);
    es = new EventSource(url);
    es.onmessage = (m) => {
      let d; try { d = JSON.parse(m.data); } catch { return; }
      if (d.type === "stage") prog.textContent = d.message || d.stage;
      else if (d.type === "hits") prog.textContent = d.hits.length + " candidates";
      else if (d.type === "pages") {
        prog.textContent = "Reading " + d.pages.length + " pages…";
        evdsum.textContent = "Evidence · " + d.pages.length + " pages";
        thumbs.innerHTML = d.pages.map((p) =>
          "<img loading='lazy' src='/img/" + p.img + "' title='" + esc(p.doc_id) + " p" + p.page_no + "'>").join("");
      }
      else if (d.type === "token") { buf += d.text; ans.textContent = buf; }
      else if (d.type === "done") done(d);
      else if (d.type === "error") done(null, d.error);
    };
    es.onerror = () => done(null, "connection lost — try again");
  }

  function done(d, e) {
    if (es) { es.close(); es = null; }
    go.disabled = false; go.textContent = "Ask";
    prog.hidden = true;
    if (e || !d) { err.hidden = false; err.textContent = e || "failed"; return; }
    ans.textContent = d.answer;
    cite.textContent = (d.cited && d.cited.length)
      ? d.cited.map((c) => c.document + " p" + c.page).join(" · ") + " · " + d.elapsed_s + "s"
      : "no citations";
    if (d.ok && thumbs.children.length) evd.open = true;
  }
})();
