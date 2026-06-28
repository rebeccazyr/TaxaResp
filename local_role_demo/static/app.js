const els = {
  status: document.getElementById("status"),
  paperSelect: document.getElementById("paperSelect"),
  paperTitle: document.getElementById("paperTitle"),
  runMeta: document.getElementById("runMeta"),
  summaryCards: document.getElementById("summaryCards"),
  taskNodes: document.getElementById("taskNodes"),
  nodeDetail: document.getElementById("nodeDetail"),
};

let currentTaskData = null;

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function pct(value) {
  const n = Number(value || 0);
  return `${(100 * n).toFixed(2)}%`;
}

function num(value, digits = 4) {
  const n = Number(value || 0);
  return n.toFixed(digits);
}

async function getJson(url) {
  const resp = await fetch(url);
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.error || `Request failed: ${url}`);
  return data;
}

function renderOverview(data) {
  els.paperSelect.innerHTML = data.tasks
    .map((task) => `<option value="${escapeHtml(task.paper_id)}">${escapeHtml(task.label)}</option>`)
    .join("");
}

function renderSummary(data) {
  const members = data.task.members
    .map((m) => `<span class="chip">${escapeHtml(m.expert_name)}</span>`)
    .join("");
  const allHitMembers = data.local_metrics.all_assignment_hit_members?.length
    ? data.local_metrics.all_assignment_hit_members
        .map((m) => `<span class="chip hit-chip">${escapeHtml(m.expert_name)}</span>`)
        .join("")
    : `<span class="chip muted-chip">none</span>`;
  els.paperTitle.textContent = data.title || `Paper ${data.paper_id}`;
  els.runMeta.textContent = `paper_id=${data.paper_id} · team_size hits=${data.local_metrics.hits_at_team_size} · all-assignment hits=${data.local_metrics.all_assignment_hits}`;
  els.summaryCards.innerHTML = `
    <article>
      <span class="metric-label">Assigned taxonomy nodes</span>
      <strong>${data.direct_nodes.length}</strong>
    </article>
    <article>
      <span class="metric-label">Hits after team_size limit</span>
      <strong>${data.local_metrics.hits_at_team_size} / ${data.task.members.length}</strong>
      <small>selected ${data.local_metrics.team_size} experts</small>
    </article>
    <article>
      <span class="metric-label">Hits from all node assignments</span>
      <strong>${data.local_metrics.all_assignment_hits} / ${data.task.members.length}</strong>
      <small>${data.local_metrics.all_assignment_selected_experts} assigned experts</small>
      <div class="chips">${allHitMembers}</div>
    </article>
    <article>
      <span class="metric-label">Ground-truth members</span>
      <div class="chips">${members}</div>
    </article>
  `;
}

function renderTaskNodes(nodes) {
  const tree = buildTree(nodes);
  els.taskNodes.innerHTML = `<ul class="taxonomy-tree">${renderTreeBranches(tree)}</ul>`;
  els.taskNodes.querySelectorAll(".tree-node").forEach((btn) => {
    btn.addEventListener("click", () => {
      const idx = Number(btn.dataset.nodeIndex);
      renderNodeDetail(nodes[idx]);
      els.taskNodes.querySelectorAll(".tree-node").forEach((x) => x.classList.remove("active"));
      btn.classList.add("active");
    });
  });
  if (nodes.length) {
    renderNodeDetail(nodes[0]);
    const first = els.taskNodes.querySelector(".tree-node");
    if (first) first.classList.add("active");
  }
}

function buildTree(nodes) {
  const byId = new Map();
  nodes.forEach((node, idx) => byId.set(node.node_id, { ...node, node_index: idx, children: [] }));

  const roots = [];
  for (const treeNode of byId.values()) {
    const parentId = treeNode.tree_parent_id;
    const parent = parentId ? byId.get(parentId) : null;
    if (parent && parent.node_id !== treeNode.node_id) {
      parent.children.push(treeNode);
    } else {
      roots.push(treeNode);
    }
  }

  const byRank = (a, b) => Number(a.bfs_rank || 0) - Number(b.bfs_rank || 0);
  const sortBranch = (branch) => {
    branch.sort(byRank);
    branch.forEach((node) => sortBranch(node.children));
  };
  sortBranch(roots);
  return roots;
}

function renderTreeBranches(nodes) {
  return nodes
    .map((node) => {
      const hasChildren = node.children.length > 0;
      return `
        <li class="tree-branch">
          ${renderTreeNode(node, hasChildren)}
          ${hasChildren ? `<ul>${renderTreeBranches(node.children)}</ul>` : ""}
        </li>
      `;
    })
    .join("");
}

function renderTreeNode(node, hasChildren) {
  const hitBadge = node.is_actual_member ? `<span class="tree-hit-badge">GT hit</span>` : "";
  return `
    <button class="tree-node ${node.node_id ? "" : "muted-row"} ${node.is_actual_member ? "hit-row" : ""}" data-node-index="${node.node_index}">
      <span class="tree-level">${hasChildren ? "▾" : "•"} L${escapeHtml(node.node_level || "0")}</span>
      <span class="tree-main">
        <strong>${escapeHtml(node.node_name)}</strong>
        <small>${escapeHtml(node.expert_name)} · weighted ${num(node.weighted_score, 4)} ${hitBadge}</small>
      </span>
      <span class="tree-score">${num(node.similarity, 3)}</span>
    </button>
  `;
}

function renderNodeDetail(node) {
  if (!node) {
    els.nodeDetail.className = "node-detail empty";
    els.nodeDetail.innerHTML = "<p>Click a taxonomy node to inspect its assignment.</p>";
    return;
  }
  els.nodeDetail.className = "node-detail";
  els.nodeDetail.innerHTML = `
    <div class="detail-head">
      <div>
        <span class="metric-label">Taxonomy node</span>
        <h4>${escapeHtml(node.node_name)}</h4>
        <p>level ${escapeHtml(node.node_level || "-")} · node ${escapeHtml(node.node_id)}</p>
      </div>
      <strong>#${escapeHtml(node.bfs_rank)}</strong>
    </div>
    <div class="detail-grid">
      <div><span>similarity</span><b>${num(node.similarity, 6)}</b></div>
      <div><span>node importance</span><b>${num(node.node_importance, 6)}</b></div>
      <div><span>weighted score</span><b>${num(node.weighted_score, 6)}</b></div>
      <div><span>subtree skills</span><b>${escapeHtml(node.subtree_skill_count)}</b></div>
    </div>
    <div class="assignment-line">
      <span>assigned expert</span>
      <strong>${escapeHtml(node.expert_name)} (${escapeHtml(node.expert_id)})</strong>
      <em>${node.is_actual_member ? "actual member" : "candidate"}</em>
    </div>
    <code>task embedding ${escapeHtml(node.embedding_id || "no embedding id")}</code>
    ${node.role_text ? `<p>${escapeHtml(node.role_text)}</p>` : ""}
  `;
}

async function loadTask(paperId) {
  els.status.textContent = "Loading precomputed flow";
  const data = await getJson(`/api/method/task?paper_id=${encodeURIComponent(paperId)}`);
  currentTaskData = data;
  renderSummary(data);
  renderTaskNodes(data.direct_nodes);
  els.status.textContent = "Ready";
}

async function init() {
  try {
    const overview = await getJson("/api/method/overview");
    renderOverview(overview);
    const first = overview.tasks[0]?.paper_id;
    if (first) await loadTask(first);
  } catch (err) {
    els.status.textContent = "Error";
    els.summaryCards.innerHTML = `<article class="error">${escapeHtml(err.message)}</article>`;
  }
}

els.paperSelect.addEventListener("change", () => loadTask(els.paperSelect.value));
init();
