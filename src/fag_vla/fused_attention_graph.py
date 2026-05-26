"""
Fused Attention Graph (FAG) — core module.

Bridges VLM-layer attention (language_attn) and Action-Expert-layer attention
(expert_attn) via hidden-state intermediate nodes, building an explicit directed
weighted graph that represents cross-modal semantic flow inside pi0.5.

Graph convention
----------------
Node IDs follow the token-index convention of ExplainVLA/settings.py:
  [0,   256)  → image1 patches
  [256, 512)  → image2 patches
  [512, 768)  → image3 patches  (optional)
  [768, 768+T) → text tokens (instruction)
  [768+T, …)  → state tokens

Edges carry weights derived from attention values after multi-head merging.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ModalSegment:
    """Defines a contiguous token range belonging to one modality."""
    name: str
    start: int
    end: int  # exclusive

    @property
    def size(self) -> int:
        return self.end - self.start

    def indices(self) -> range:
        return range(self.start, self.end)


@dataclass
class FAGConfig:
    """Configuration for FAG construction."""
    segments: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    layer_agg: str = "mean"          # "mean" | "sum" | "max" | "last"
    head_merge: str = "mean"         # "mean" | "max" | "sum"
    edge_threshold: float = 1e-4     # prune edges below this weight
    cross_model_bridge: bool = True  # use hidden-state bridge for cross-model fusion
    include_image3: bool = False


# ---------------------------------------------------------------------------
# Node / edge helper utilities
# ---------------------------------------------------------------------------

def _node_label(token_idx: int, segments: Dict[str, Tuple[int, int]]) -> str:
    """Return human-readable node label from global token index."""
    for name, (s, e) in segments.items():
        if s <= token_idx < e:
            local = token_idx - s
            return f"{name}[{local}]"
    return f"tok[{token_idx}]"


def _merge_heads(attn: torch.Tensor, strategy: str = "mean") -> torch.Tensor:
    """
    attn: (num_heads, q_len, k_len) → (q_len, k_len)
    """
    if attn.dim() == 4:
        attn = attn.squeeze(0)
    if attn.dim() == 3:
        if strategy == "mean":
            return attn.mean(0)
        elif strategy == "max":
            return attn.max(0).values
        elif strategy == "sum":
            return attn.sum(0)
    return attn  # already 2-D


def _aggregate_layers(
    layer_attn_dict: Dict[int, torch.Tensor],
    strategy: str = "mean",
) -> torch.Tensor:
    """
    Aggregate attention matrices across transformer layers.

    layer_attn_dict: {layer_idx: Tensor(1, heads, seq, seq) or (heads, seq, seq)}
    Returns: Tensor (seq, seq)
    """
    tensors = []
    for layer_idx in sorted(layer_attn_dict.keys()):
        t = layer_attn_dict[layer_idx]
        if isinstance(t, torch.Tensor):
            t = t.float()
            # squeeze batch dim if present
            if t.dim() == 4:
                t = t.squeeze(0)
            # merge heads → (seq, seq)
            t_merged = _merge_heads(t, strategy)
            tensors.append(t_merged)

    if not tensors:
        return None

    stacked = torch.stack(tensors, dim=0)  # (L, seq, seq)
    if strategy == "mean":
        return stacked.mean(0)
    elif strategy == "max":
        return stacked.max(0).values
    elif strategy == "sum":
        return stacked.sum(0)
    elif strategy == "last":
        return stacked[-1]
    return stacked.mean(0)


# ---------------------------------------------------------------------------
# FAG builder
# ---------------------------------------------------------------------------

class FusedAttentionGraph:
    """
    Builds a fused directed weighted graph from VLM + Action-Expert attention data.

    Usage
    -----
    fag = FusedAttentionGraph(config)
    G   = fag.build(
            vlm_layer_attn   = tracer.language_attn[step],     # {layer: Tensor}
            expert_step_attn = tracer.expert_attn[step][t],    # {layer: Tensor}
          )
    adj = fag.to_adjacency_matrix(G, src_modal="text", tgt_modal="image1")
    """

    def __init__(self, config: FAGConfig):
        self.config = config
        self._segments: Dict[str, ModalSegment] = {
            name: ModalSegment(name, s, e)
            for name, (s, e) in config.segments.items()
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        vlm_layer_attn: Dict[int, torch.Tensor],
        expert_step_attn: Optional[Dict[int, torch.Tensor]] = None,
        hidden_states: Optional[Dict[str, torch.Tensor]] = None,
    ) -> nx.DiGraph:
        """
        Build the Fused Attention Graph.

        Parameters
        ----------
        vlm_layer_attn:   {layer_idx: (1, heads, seq, seq)}  — from ATTENTION_TRACER.language_attn
        expert_step_attn: {layer_idx: (1, heads, seq2, seq2)} — from expert_attn[step][timestep]
        hidden_states:    optional dict with keys 'vlm', 'expert' containing last hidden state tensors
                          used as cross-model bridge when cross_model_bridge=True

        Returns
        -------
        nx.DiGraph with node attributes {modal, local_idx, label}
                         edge attributes {weight, source}
        """
        G = nx.DiGraph()
        self._add_modal_nodes(G)

        # --- VLM intra-model attention ---
        if vlm_layer_attn:
            vlm_agg = _aggregate_layers(vlm_layer_attn, self.config.layer_agg)
            if vlm_agg is not None:
                self._add_attention_edges(G, vlm_agg, source_tag="vlm")

        # --- Expert intra-model attention ---
        if expert_step_attn:
            exp_agg = _aggregate_layers(expert_step_attn, self.config.layer_agg)
            if exp_agg is not None:
                # Expert seq may differ from VLM seq — only write expert nodes if sizes align
                self._add_attention_edges(G, exp_agg, source_tag="expert",
                                          offset=0, max_nodes=exp_agg.shape[0])

        # --- Cross-model bridge edges via hidden states ---
        if self.config.cross_model_bridge and hidden_states is not None:
            self._add_bridge_edges(G, hidden_states)

        return G

    def build_temporal_sequence(
        self,
        vlm_layer_attn: Dict[int, torch.Tensor],
        expert_attn_over_time: Dict[int, Dict[int, torch.Tensor]],
    ) -> List[nx.DiGraph]:
        """
        Build a time-ordered list of FAGs, one per diffusion timestep.

        expert_attn_over_time: {timestep: {layer: Tensor}}
        Returns: list of DiGraph ordered by ascending timestep
        """
        graphs = []
        for ts in sorted(expert_attn_over_time.keys()):
            G = self.build(vlm_layer_attn, expert_attn_over_time[ts])
            G.graph["timestep"] = ts
            graphs.append(G)
        return graphs

    def to_adjacency_matrix(
        self,
        G: nx.DiGraph,
        src_modal: str,
        tgt_modal: str,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Extract the attention sub-matrix from src_modal nodes to tgt_modal nodes.

        Returns np.ndarray of shape (|src|, |tgt|).
        """
        src_seg = self._segments.get(src_modal)
        tgt_seg = self._segments.get(tgt_modal)
        if src_seg is None or tgt_seg is None:
            raise ValueError(f"Unknown modal segment: {src_modal} or {tgt_modal}")

        mat = np.zeros((src_seg.size, tgt_seg.size), dtype=np.float32)
        for i, src_idx in enumerate(src_seg.indices()):
            for j, tgt_idx in enumerate(tgt_seg.indices()):
                if G.has_edge(src_idx, tgt_idx):
                    mat[i, j] = G[src_idx][tgt_idx].get("weight", 0.0)

        if normalize and mat.max() > 0:
            mat = mat / mat.max()
        return mat

    def graph_metrics(self, G: nx.DiGraph) -> Dict[str, float]:
        """
        Compute basic graph-theoretic metrics on the FAG.

        Returns dict with keys:
          - num_nodes, num_edges
          - density
          - in_degree_{modal}_mean  (for each segment)
          - out_degree_{modal}_mean
          - attention_entropy_{modal → modal}  (Shannon entropy of edge weights)
        """
        metrics: Dict[str, float] = {
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "density": nx.density(G),
        }

        for modal, seg in self._segments.items():
            nodes_in_modal = [n for n in G.nodes if G.nodes[n].get("modal") == modal]
            if nodes_in_modal:
                in_d = [G.in_degree(n) for n in nodes_in_modal]
                out_d = [G.out_degree(n) for n in nodes_in_modal]
                metrics[f"in_degree_{modal}_mean"] = float(np.mean(in_d))
                metrics[f"out_degree_{modal}_mean"] = float(np.mean(out_d))

        # Edge weight entropy per cross-modal pair
        for src_modal in self._segments:
            for tgt_modal in self._segments:
                if src_modal == tgt_modal:
                    continue
                src_seg = self._segments[src_modal]
                tgt_seg = self._segments[tgt_modal]
                weights = [
                    G[u][v]["weight"]
                    for u in src_seg.indices()
                    for v in tgt_seg.indices()
                    if G.has_edge(u, v)
                ]
                if weights:
                    w = np.array(weights, dtype=np.float64)
                    w = w / (w.sum() + 1e-12)
                    entropy = float(-np.sum(w * np.log(w + 1e-12)))
                    metrics[f"entropy_{src_modal}_to_{tgt_modal}"] = entropy

        return metrics

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _add_modal_nodes(self, G: nx.DiGraph):
        for name, seg in self._segments.items():
            if name == "image3" and not self.config.include_image3:
                continue
            for tok_idx in seg.indices():
                G.add_node(tok_idx, modal=name, local_idx=tok_idx - seg.start,
                           label=f"{name}[{tok_idx - seg.start}]")

    def _add_attention_edges(
        self,
        G: nx.DiGraph,
        agg_attn: torch.Tensor,
        source_tag: str,
        offset: int = 0,
        max_nodes: Optional[int] = None,
    ):
        """
        Add directed edges u→v with weight = agg_attn[u-offset, v-offset].
        Only adds edges between nodes already in G.
        """
        attn_np = agg_attn.detach().cpu().float().numpy()
        nodes = sorted(G.nodes())
        if max_nodes is not None:
            nodes = [n for n in nodes if n - offset < max_nodes]

        threshold = self.config.edge_threshold

        for u in nodes:
            ui = u - offset
            if ui < 0 or ui >= attn_np.shape[0]:
                continue
            row = attn_np[ui]
            for v in nodes:
                vi = v - offset
                if vi < 0 or vi >= attn_np.shape[1]:
                    continue
                w = float(row[vi])
                if w < threshold:
                    continue
                if G.has_edge(u, v):
                    # Fuse: take max of existing and new weight
                    G[u][v]["weight"] = max(G[u][v]["weight"], w)
                    G[u][v]["sources"].add(source_tag)
                else:
                    G.add_edge(u, v, weight=w, sources={source_tag})

    def _add_bridge_edges(
        self,
        G: nx.DiGraph,
        hidden_states: Dict[str, torch.Tensor],
    ):
        """
        Add cross-model bridge edges using cosine similarity between hidden states
        of VLM output tokens and Expert input tokens.
        """
        vlm_hs = hidden_states.get("vlm")    # (seq_vlm, dim)
        exp_hs = hidden_states.get("expert") # (seq_exp, dim)
        if vlm_hs is None or exp_hs is None:
            return

        vlm_hs = vlm_hs.float()
        exp_hs = exp_hs.float()

        # Normalise for cosine similarity
        vlm_norm = torch.nn.functional.normalize(vlm_hs, dim=-1)
        exp_norm = torch.nn.functional.normalize(exp_hs, dim=-1)
        # (seq_vlm, seq_exp) cosine similarity matrix
        sim = (vlm_norm @ exp_norm.T).detach().cpu().numpy()

        threshold = self.config.edge_threshold
        seg_names = list(self._segments.keys())

        for ui, u_node in enumerate(sorted(G.nodes())):
            if ui >= sim.shape[0]:
                break
            for vi, v_node in enumerate(sorted(G.nodes())):
                if vi >= sim.shape[1]:
                    break
                w = float(sim[ui, vi])
                if w < threshold:
                    continue
                if G.has_edge(u_node, v_node):
                    G[u_node][v_node]["weight"] = (G[u_node][v_node]["weight"] + w) / 2.0
                    G[u_node][v_node]["sources"].add("bridge")
                else:
                    G.add_edge(u_node, v_node, weight=w, sources={"bridge"})


# ---------------------------------------------------------------------------
# Convergence analysis helper
# ---------------------------------------------------------------------------

def analyse_temporal_convergence(
    graph_sequence: List[nx.DiGraph],
    src_modal: str,
    tgt_modal: str,
    fag: FusedAttentionGraph,
) -> Dict[str, np.ndarray]:
    """
    Track how attention from src_modal to tgt_modal evolves across diffusion steps.

    Returns dict:
      'timesteps':      (T,) int array
      'mean_weight':    (T,) average edge weight
      'entropy':        (T,) Shannon entropy of attention distribution
      'adjacency':      (T, |src|, |tgt|) stacked adjacency matrices
    """
    timesteps = []
    mean_weights = []
    entropies = []
    adjs = []

    for G in graph_sequence:
        ts = G.graph.get("timestep", len(timesteps))
        timesteps.append(ts)

        adj = fag.to_adjacency_matrix(G, src_modal, tgt_modal, normalize=True)
        adjs.append(adj)
        mean_weights.append(float(adj.mean()))

        flat = adj.flatten() + 1e-12
        flat /= flat.sum()
        entropy = float(-np.sum(flat * np.log(flat)))
        entropies.append(entropy)

    return {
        "timesteps": np.array(timesteps, dtype=np.int32),
        "mean_weight": np.array(mean_weights, dtype=np.float32),
        "entropy": np.array(entropies, dtype=np.float32),
        "adjacency": np.stack(adjs, axis=0),
    }
