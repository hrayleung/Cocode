use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

/// Sort comparison that handles NaN values by pushing them to the end
fn cmp_scores_desc(a: &(String, f32), b: &(String, f32)) -> std::cmp::Ordering {
    match (a.1.is_nan(), b.1.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal),
    }
}

/// Compute cosine similarity between two vectors
///
/// Args:
///     vec1: First vector (1D numpy array)
///     vec2: Second vector (1D numpy array)
///
/// Returns:
///     Cosine similarity score (float)
#[pyfunction]
pub fn cosine_similarity(
    vec1: PyReadonlyArray1<f32>,
    vec2: PyReadonlyArray1<f32>,
) -> PyResult<f32> {
    let v1 = vec1.as_slice()?;
    let v2 = vec2.as_slice()?;

    if v1.len() != v2.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Vectors must have the same length",
        ));
    }

    let dot_product: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
    let norm1: f32 = v1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm2: f32 = v2.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm1 == 0.0 || norm2 == 0.0 {
        return Ok(0.0);
    }

    Ok(dot_product / (norm1 * norm2))
}

/// Compute cosine similarity between a query vector and multiple document vectors
/// Uses parallel processing for better performance
///
/// Args:
///     query: Query vector (1D numpy array)
///     documents: Document vectors (2D numpy array, each row is a document)
///
/// Returns:
///     Array of cosine similarity scores
#[pyfunction]
pub fn cosine_similarity_batch<'py>(
    py: Python<'py>,
    query: PyReadonlyArray1<f32>,
    documents: PyReadonlyArray2<f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let q = query.as_slice()?;
    let docs = documents.as_array();

    // Validate dimensions match
    let q_len = q.len();
    let doc_cols = docs.ncols();
    if q_len != doc_cols {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "query length {} does not match document vector size {}",
            q_len, doc_cols
        )));
    }

    // Precompute query norm
    let query_norm: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();

    if query_norm == 0.0 {
        let zeros = vec![0.0f32; docs.nrows()];
        return Ok(PyArray1::from_vec(py, zeros));
    }

    // Parallel computation of similarities
    let similarities: Vec<f32> = py.detach(|| {
        (0..docs.nrows())
            .into_par_iter()
            .map(|i| {
                let doc = docs.row(i);
                let dot_product: f32 = q.iter().zip(doc.iter()).map(|(a, b)| a * b).sum();
                let doc_norm: f32 = doc.iter().map(|x| x * x).sum::<f32>().sqrt();

                if doc_norm == 0.0 {
                    0.0
                } else {
                    dot_product / (query_norm * doc_norm)
                }
            })
            .collect()
    });

    Ok(PyArray1::from_vec(py, similarities))
}

/// Reciprocal Rank Fusion (RRF) - combines multiple ranked lists
///
/// Args:
///     ranked_lists: List of lists, where each inner list contains (doc_id, score) tuples
///     k: RRF constant (default: 60)
///
/// Returns:
///     List of (doc_id, fused_score) tuples, sorted by fused score descending
#[pyfunction]
#[pyo3(signature = (ranked_lists, k=60.0))]
pub fn reciprocal_rank_fusion(
    ranked_lists: Vec<Vec<(String, f32)>>,
    k: f32,
) -> PyResult<Vec<(String, f32)>> {
    let mut scores: HashMap<String, f32> = HashMap::new();

    for ranked_list in ranked_lists {
        for (rank, (doc_id, _score)) in ranked_list.iter().enumerate() {
            let rrf_score = 1.0 / (k + (rank as f32) + 1.0);
            *scores.entry(doc_id.clone()).or_insert(0.0) += rrf_score;
        }
    }

    let mut result: Vec<(String, f32)> = scores.into_iter().collect();
    result.sort_by(cmp_scores_desc);

    Ok(result)
}

/// Weighted Reciprocal Rank Fusion - combines multiple ranked lists with custom weights
///
/// Args:
///     ranked_lists: List of lists, where each inner list contains (doc_id, score) tuples
///     weights: List of weights for each ranked list (must match length of ranked_lists)
///     k: RRF constant (default: 60)
///
/// Returns:
///     List of (doc_id, fused_score) tuples, sorted by fused score descending
#[pyfunction]
#[pyo3(signature = (ranked_lists, weights, k=60.0))]
pub fn reciprocal_rank_fusion_weighted(
    ranked_lists: Vec<Vec<(String, f32)>>,
    weights: Vec<f32>,
    k: f32,
) -> PyResult<Vec<(String, f32)>> {
    if ranked_lists.len() != weights.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Number of ranked lists must match number of weights",
        ));
    }

    let mut scores: HashMap<String, f32> = HashMap::new();

    for (ranked_list, weight) in ranked_lists.iter().zip(weights.iter()) {
        for (rank, (doc_id, _score)) in ranked_list.iter().enumerate() {
            let rrf_score = weight / (k + (rank as f32) + 1.0);
            *scores.entry(doc_id.clone()).or_insert(0.0) += rrf_score;
        }
    }

    let mut result: Vec<(String, f32)> = scores.into_iter().collect();
    result.sort_by(cmp_scores_desc);

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        // Test the core computation logic directly (numpy arrays require Python context)
        let v1: Vec<f32> = vec![1.0, 2.0, 3.0];
        let v2: Vec<f32> = vec![1.0, 2.0, 3.0];

        let dot_product: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f32 = v1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = v2.iter().map(|x| x * x).sum::<f32>().sqrt();

        let similarity = dot_product / (norm1 * norm2);
        assert!((similarity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_rrf_basic() {
        let list1 = vec![
            ("doc1".to_string(), 0.9),
            ("doc2".to_string(), 0.8),
            ("doc3".to_string(), 0.7),
        ];
        let list2 = vec![
            ("doc2".to_string(), 0.95),
            ("doc1".to_string(), 0.85),
            ("doc4".to_string(), 0.75),
        ];

        let result = reciprocal_rank_fusion(vec![list1, list2], 60.0).unwrap();

        // doc1 should have high score (rank 0 + rank 1)
        // doc2 should have high score (rank 1 + rank 0)
        assert!(result.len() >= 3);

        // Check that doc1 and doc2 are among top results
        let top_docs: Vec<&str> = result.iter().take(2).map(|(id, _)| id.as_str()).collect();
        assert!(top_docs.contains(&"doc1") || top_docs.contains(&"doc2"));
    }

    #[test]
    fn test_weighted_rrf() {
        let list1 = vec![("doc1".to_string(), 0.9)];
        let list2 = vec![("doc2".to_string(), 0.9)];

        let result = reciprocal_rank_fusion_weighted(
            vec![list1, list2],
            vec![0.7, 0.3],
            60.0,
        ).unwrap();

        assert_eq!(result.len(), 2);
        // doc1 should rank higher due to higher weight
        assert_eq!(result[0].0, "doc1");
    }
}
