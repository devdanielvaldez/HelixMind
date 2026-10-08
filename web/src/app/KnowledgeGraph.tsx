"use client";

import { useMemo, useState } from "react";
import {
  Background,
  Controls,
  MiniMap,
  MarkerType,
  ReactFlow,
  type Edge,
  type Node,
} from "@xyflow/react";

type Graph = {
  nodes: { id: string; kind: string; label: string }[];
  edges: { source: string; relation: string; target: string; confidence: number; evidence_ids: string; artifact_ids: string }[];
};

export default function KnowledgeGraph({ graph }: { graph: Graph }) {
  const [query, setQuery] = useState("");
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const normalizedQuery = query.trim().toLocaleLowerCase();
  const visibleGraph = useMemo(() => {
    const matched = graph.nodes.filter((node) => !normalizedQuery ||
      `${node.id} ${node.label} ${node.kind}`.toLocaleLowerCase().includes(normalizedQuery));
    const matchedIds = new Set(matched.map((node) => node.id));
    const related = normalizedQuery
      ? graph.edges.flatMap((edge) => matchedIds.has(edge.source) ? [edge.target] : matchedIds.has(edge.target) ? [edge.source] : [])
      : [];
    const visibleIds = new Set([...matchedIds, ...related]);
    const rows = graph.nodes.filter((node) => visibleIds.has(node.id));
    const columns = Math.max(1, Math.ceil(Math.sqrt(rows.length)));
    const nodes: Node[] = rows.map((node, index) => {
      return {
        id: node.id,
        type: "default",
        position: { x: (index % columns) * 250, y: Math.floor(index / columns) * 150 },
        data: { label: `${node.kind}: ${node.label}` },
      };
    });
    const edges: Edge[] = graph.edges
      .filter((edge) => visibleIds.has(edge.source) && visibleIds.has(edge.target))
      .map((edge, index) => ({
        id: `${edge.source}-${edge.relation}-${edge.target}-${index}`,
        source: edge.source,
        target: edge.target,
        label: edge.relation,
        type: "smoothstep",
        markerEnd: { type: MarkerType.ArrowClosed, color: "#4a7868" },
        style: { stroke: "#82a997", strokeWidth: 1.5 },
        labelStyle: { fill: "#405d57", fontSize: 10 },
        data: { confidence: edge.confidence, evidenceIds: edge.evidence_ids, artifactIds: edge.artifact_ids },
      }));
    return { nodes, edges, rows };
  }, [graph, normalizedQuery]);

  const selected = graph.nodes.find((node) => node.id === selectedNode);
  const selectedRelations = selected ? graph.edges.filter((edge) => edge.source === selected.id || edge.target === selected.id) : [];

  return <div className="knowledge-graph-explorer">
    <div className="graph-toolbar">
      <label htmlFor="knowledge-graph-search">Find an entity</label>
      <input id="knowledge-graph-search" type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search label, type or stable ID" />
      <span>{visibleGraph.rows.length} of {graph.nodes.length} entities · {visibleGraph.edges.length} visible relationships</span>
    </div>
    <div className="knowledge-graph-canvas" aria-label="Interactive biomedical knowledge graph">
      <ReactFlow
        key={normalizedQuery || "all-entities"}
        nodes={visibleGraph.nodes}
        edges={visibleGraph.edges}
        fitView
        fitViewOptions={{ padding: 0.2, maxZoom: 1 }}
        minZoom={0.02}
        maxZoom={1.5}
        nodesConnectable={false}
        onNodeClick={(_, node) => setSelectedNode(node.id)}
        proOptions={{ hideAttribution: true }}
      >
        <Background color="#dce8e2" gap={22} />
        <MiniMap pannable zoomable nodeColor="#91bfaa" />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
    {selected && <section className="graph-selection" aria-live="polite">
      <div><span className="graph-kind">{selected.kind}</span><b>{selected.label}</b><code>{selected.id}</code></div>
      <button type="button" onClick={() => setSelectedNode(null)}>Clear selection</button>
      {selectedRelations.length === 0 ? <p>No source-linked relationships are recorded for this entity.</p> : <ul>{selectedRelations.map((edge, index) => <li key={`${edge.source}-${edge.relation}-${edge.target}-${index}`}>
        <b>{edge.source === selected.id ? "Outgoing" : "Incoming"}</b> · {edge.source} <span>{edge.relation}</span> {edge.target} · annotation {edge.confidence.toFixed(2)} · evidence {edge.evidence_ids || "none"} · source artifacts {edge.artifact_ids || "none"}
      </li>)}</ul>}
    </section>}
    {visibleGraph.nodes.length === 0 && <p className="graph-no-results">No entities match this search.</p>}
  </div>;
}
