use pyo3::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};
use petgraph::graph::{DiGraph, NodeIndex};
use ahash::AHashMap;

/// Compute PageRank scores for a directed graph
///
/// Args:
///     edges: List of (source, target) tuples representing directed edges
///     damping: Damping factor (default: 0.85)
///     max_iterations: Maximum number of iterations (default: 100)
///     tolerance: Convergence tolerance (default: 1e-6)
///
/// Returns:
///     Dictionary mapping node names to PageRank scores
#[pyfunction]
#[pyo3(signature = (edges, damping=0.85, max_iterations=100, tolerance=1e-6))]
pub fn pagerank(
    edges: Vec<(String, String)>,
    damping: f32,
    max_iterations: usize,
    tolerance: f32,
) -> PyResult<HashMap<String, f32>> {
    if edges.is_empty() {
        return Ok(HashMap::new());
    }

    // Build node index mapping
    let mut node_to_idx: AHashMap<String, usize> = AHashMap::new();
    let mut idx_to_node: Vec<String> = Vec::new();

    for (source, target) in &edges {
        if !node_to_idx.contains_key(source) {
            node_to_idx.insert(source.clone(), idx_to_node.len());
            idx_to_node.push(source.clone());
        }
        if !node_to_idx.contains_key(target) {
            node_to_idx.insert(target.clone(), idx_to_node.len());
            idx_to_node.push(target.clone());
        }
    }

    let num_nodes = idx_to_node.len();

    // Build adjacency list and out-degree count
    let mut out_edges: Vec<Vec<usize>> = vec![Vec::new(); num_nodes];
    let mut out_degree: Vec<usize> = vec![0; num_nodes];

    for (source, target) in &edges {
        let src_idx = node_to_idx[source];
        let tgt_idx = node_to_idx[target];
        out_edges[src_idx].push(tgt_idx);
        out_degree[src_idx] += 1;
    }

    // Initialize PageRank scores
    let initial_score = 1.0 / num_nodes as f32;
    let mut scores: Vec<f32> = vec![initial_score; num_nodes];
    let mut new_scores: Vec<f32> = vec![0.0; num_nodes];

    // Power iteration
    for _ in 0..max_iterations {
        // Calculate new scores
        for i in 0..num_nodes {
            new_scores[i] = (1.0 - damping) / num_nodes as f32;
        }

        for src_idx in 0..num_nodes {
            if out_degree[src_idx] > 0 {
                let contribution = damping * scores[src_idx] / out_degree[src_idx] as f32;
                for &tgt_idx in &out_edges[src_idx] {
                    new_scores[tgt_idx] += contribution;
                }
            } else {
                // Dangling node: distribute score evenly
                let contribution = damping * scores[src_idx] / num_nodes as f32;
                for i in 0..num_nodes {
                    new_scores[i] += contribution;
                }
            }
        }

        // Check convergence
        let mut diff = 0.0;
        for i in 0..num_nodes {
            diff += (new_scores[i] - scores[i]).abs();
        }

        scores.copy_from_slice(&new_scores);

        if diff < tolerance {
            break;
        }
    }

    // Convert to HashMap
    let mut result = HashMap::new();
    for (idx, node) in idx_to_node.iter().enumerate() {
        result.insert(node.clone(), scores[idx]);
    }

    Ok(result)
}

/// Multi-hop BFS graph expansion with bidirectional traversal
///
/// Args:
///     edges: List of (source, target) tuples representing directed edges
///     start_nodes: List of starting node names
///     max_hops: Maximum number of hops to traverse (default: 3)
///     max_results: Maximum number of nodes to return (default: 30)
///     bidirectional: Whether to traverse both forward and backward (default: true)
///
/// Returns:
///     Dictionary mapping node names to their hop distance from start nodes
#[pyfunction]
#[pyo3(signature = (edges, start_nodes, max_hops=3, max_results=30, bidirectional=true))]
pub fn bfs_expansion(
    edges: Vec<(String, String)>,
    start_nodes: Vec<String>,
    max_hops: usize,
    max_results: usize,
    bidirectional: bool,
) -> PyResult<HashMap<String, usize>> {
    if edges.is_empty() || start_nodes.is_empty() {
        return Ok(HashMap::new());
    }

    // Build adjacency lists
    let mut forward_edges: AHashMap<String, Vec<String>> = AHashMap::new();
    let mut backward_edges: AHashMap<String, Vec<String>> = AHashMap::new();

    for (source, target) in edges {
        forward_edges
            .entry(source.clone())
            .or_insert_with(Vec::new)
            .push(target.clone());

        if bidirectional {
            backward_edges
                .entry(target)
                .or_insert_with(Vec::new)
                .push(source);
        }
    }

    // BFS with hop tracking
    let mut visited: HashSet<String> = HashSet::new();
    let mut hop_distances: HashMap<String, usize> = HashMap::new();
    let mut queue: VecDeque<(String, usize)> = VecDeque::new();

    // Early return if max_results is 0
    if max_results == 0 {
        return Ok(HashMap::new());
    }

    // Initialize with start nodes, respecting max_results limit
    for node in start_nodes {
        if hop_distances.len() >= max_results {
            break;
        }
        if !visited.contains(&node) {
            visited.insert(node.clone());
            hop_distances.insert(node.clone(), 0);
            queue.push_back((node, 0));
        }
    }

    // BFS traversal
    while let Some((current, hop)) = queue.pop_front() {
        // Stop if we've hit max_results
        if hop_distances.len() >= max_results {
            break;
        }

        // Skip expanding neighbors if next hop would exceed max_hops
        if hop >= max_hops {
            continue;
        }

        let next_hop = hop + 1;

        // Traverse forward edges
        if let Some(neighbors) = forward_edges.get(&current) {
            for neighbor in neighbors {
                if hop_distances.len() >= max_results {
                    break;
                }
                if !visited.contains(neighbor) {
                    visited.insert(neighbor.clone());
                    hop_distances.insert(neighbor.clone(), next_hop);
                    queue.push_back((neighbor.clone(), next_hop));
                }
            }
        }

        // Traverse backward edges if bidirectional
        if bidirectional {
            if let Some(neighbors) = backward_edges.get(&current) {
                for neighbor in neighbors {
                    if hop_distances.len() >= max_results {
                        break;
                    }
                    if !visited.contains(neighbor) {
                        visited.insert(neighbor.clone());
                        hop_distances.insert(neighbor.clone(), next_hop);
                        queue.push_back((neighbor.clone(), next_hop));
                    }
                }
            }
        }
    }

    Ok(hop_distances)
}

/// Multi-hop BFS traversal that returns predecessor edges in BFS tree order.
///
/// This is designed for graph expansion in Python where we want concrete edges:
/// (source_file, target_file, relation_type, hop_distance).
///
/// relation_type is:
/// - "imports" when traversing a forward edge (source -> target)
/// - "imported_by" when traversing a reverse edge (target <- source)
#[pyfunction]
#[pyo3(signature = (edges, start_nodes, max_hops=3, max_results=30, bidirectional=true))]
pub fn bfs_traversal_edges(
    edges: Vec<(String, String)>,
    start_nodes: Vec<String>,
    max_hops: usize,
    max_results: usize,
    bidirectional: bool,
) -> PyResult<Vec<(String, String, String, usize)>> {
    if edges.is_empty() || start_nodes.is_empty() || max_results == 0 || max_hops == 0 {
        return Ok(Vec::new());
    }

    // Build adjacency lists.
    let mut forward_edges: AHashMap<String, Vec<String>> = AHashMap::new();
    let mut backward_edges: AHashMap<String, Vec<String>> = AHashMap::new();

    for (source, target) in edges {
        forward_edges
            .entry(source.clone())
            .or_insert_with(Vec::new)
            .push(target.clone());

        if bidirectional {
            backward_edges
                .entry(target)
                .or_insert_with(Vec::new)
                .push(source);
        }
    }

    // Sort neighbors for deterministic traversal.
    for neighbors in forward_edges.values_mut() {
        neighbors.sort();
    }
    for neighbors in backward_edges.values_mut() {
        neighbors.sort();
    }

    let mut visited: HashSet<String> = HashSet::new();
    let mut queue: VecDeque<(String, usize, Option<String>, Option<String>)> = VecDeque::new();
    let mut results: Vec<(String, String, String, usize)> = Vec::new();

    // Seed BFS.
    for node in start_nodes {
        if visited.insert(node.clone()) {
            queue.push_back((node, 0, None, None));
        }
    }
    let start_count = visited.len();

    while let Some((current, hop, parent, relation)) = queue.pop_front() {
        if results.len() >= max_results {
            break;
        }

        if let (Some(p), Some(rel)) = (parent, relation) {
            results.push((p, current.clone(), rel, hop));
            if results.len() >= max_results {
                break;
            }
        }

        if hop >= max_hops {
            continue;
        }

        let next_hop = hop + 1;

        // Forward traversal: current imports neighbor.
        if let Some(neighbors) = forward_edges.get(&current) {
            for neighbor in neighbors {
                if results.len() >= max_results
                    || visited.len().saturating_sub(start_count) >= max_results
                {
                    break;
                }
                if visited.insert(neighbor.clone()) {
                    queue.push_back((
                        neighbor.clone(),
                        next_hop,
                        Some(current.clone()),
                        Some("imports".to_string()),
                    ));
                }
            }
        }

        // Backward traversal: current is imported by neighbor.
        if bidirectional {
            if let Some(neighbors) = backward_edges.get(&current) {
                for neighbor in neighbors {
                    if results.len() >= max_results
                        || visited.len().saturating_sub(start_count) >= max_results
                    {
                        break;
                    }
                    if visited.insert(neighbor.clone()) {
                        queue.push_back((
                            neighbor.clone(),
                            next_hop,
                            Some(current.clone()),
                            Some("imported_by".to_string()),
                        ));
                    }
                }
            }
        }
    }

    Ok(results)
}

/// Detect strongly connected components using Kosaraju's algorithm
///
/// Args:
///     edges: List of (source, target) tuples representing directed edges
///
/// Returns:
///     List of lists, where each inner list contains nodes in a strongly connected component
#[pyfunction]
pub fn strongly_connected_components(
    edges: Vec<(String, String)>,
) -> PyResult<Vec<Vec<String>>> {
    if edges.is_empty() {
        return Ok(Vec::new());
    }

    // Build graph using petgraph
    let mut graph = DiGraph::<String, ()>::new();
    let mut node_indices: AHashMap<String, NodeIndex> = AHashMap::new();

    // Add nodes
    for (source, target) in &edges {
        if !node_indices.contains_key(source) {
            let idx = graph.add_node(source.clone());
            node_indices.insert(source.clone(), idx);
        }
        if !node_indices.contains_key(target) {
            let idx = graph.add_node(target.clone());
            node_indices.insert(target.clone(), idx);
        }
    }

    // Add edges
    for (source, target) in edges {
        let src_idx = node_indices[&source];
        let tgt_idx = node_indices[&target];
        graph.add_edge(src_idx, tgt_idx, ());
    }

    // Compute SCCs using petgraph's algorithm
    let sccs = petgraph::algo::kosaraju_scc(&graph);

    // Convert NodeIndex back to node names
    let result: Vec<Vec<String>> = sccs
        .into_iter()
        .map(|scc| {
            scc.into_iter()
                .map(|idx| graph[idx].clone())
                .collect()
        })
        .collect();

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pagerank_simple() {
        let edges = vec![
            ("A".to_string(), "B".to_string()),
            ("B".to_string(), "C".to_string()),
            ("C".to_string(), "A".to_string()),
        ];

        let scores = pagerank(edges, 0.85, 100, 1e-6).unwrap();

        // All nodes should have similar scores in a cycle
        assert_eq!(scores.len(), 3);
        assert!(scores["A"] > 0.0);
        assert!(scores["B"] > 0.0);
        assert!(scores["C"] > 0.0);
    }

    #[test]
    fn test_pagerank_hub() {
        let edges = vec![
            ("A".to_string(), "B".to_string()),
            ("C".to_string(), "B".to_string()),
            ("D".to_string(), "B".to_string()),
        ];

        let scores = pagerank(edges, 0.85, 100, 1e-6).unwrap();

        // B should have highest score as it's pointed to by all others
        assert!(scores["B"] > scores["A"]);
        assert!(scores["B"] > scores["C"]);
        assert!(scores["B"] > scores["D"]);
    }

    #[test]
    fn test_bfs_expansion() {
        let edges = vec![
            ("A".to_string(), "B".to_string()),
            ("B".to_string(), "C".to_string()),
            ("C".to_string(), "D".to_string()),
            ("A".to_string(), "E".to_string()),
        ];

        let result = bfs_expansion(
            edges,
            vec!["A".to_string()],
            2,
            10,
            false,
        ).unwrap();

        // Should find nodes within 2 hops
        assert_eq!(result["A"], 0);
        assert_eq!(result["B"], 1);
        assert_eq!(result["E"], 1);
        assert_eq!(result["C"], 2);
    }

    #[test]
    fn test_bfs_bidirectional() {
        let edges = vec![
            ("A".to_string(), "B".to_string()),
            ("B".to_string(), "C".to_string()),
        ];

        let result = bfs_expansion(
            edges,
            vec!["B".to_string()],
            1,
            10,
            true,
        ).unwrap();

        // Should find both forward (C) and backward (A) neighbors
        assert_eq!(result["B"], 0);
        assert_eq!(result["A"], 1); // backward
        assert_eq!(result["C"], 1); // forward
    }

    #[test]
    fn test_scc() {
        let edges = vec![
            ("A".to_string(), "B".to_string()),
            ("B".to_string(), "C".to_string()),
            ("C".to_string(), "A".to_string()),
            ("D".to_string(), "E".to_string()),
        ];

        let sccs = strongly_connected_components(edges).unwrap();

        // Should find one SCC with A, B, C and separate components for D, E
        assert!(sccs.len() >= 2);

        // Find the SCC containing A, B, C
        let abc_scc = sccs.iter().find(|scc| scc.contains(&"A".to_string()));
        assert!(abc_scc.is_some());
        assert_eq!(abc_scc.unwrap().len(), 3);
    }
}
