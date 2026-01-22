//! Utility functions for hashing and similarity calculations.

use pyo3::prelude::*;
use sha2::{Digest, Sha256};

use ahash::AHashSet;
use pyo3::types::PyDict;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Component, Path, PathBuf};

/// Compute truncated SHA256 hash of content (16 hex chars).
#[pyfunction]
pub fn compute_file_hash(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    let result = hasher.finalize();
    // Take first 8 bytes (16 hex chars)
    hex::encode(&result[..8])
}

/// Compute Jaccard similarity between two strings (word-level).
#[pyfunction]
pub fn jaccard_similarity(text1: &str, text2: &str) -> f64 {
    let words1: AHashSet<&str> = text1.split_whitespace().collect();
    let words2: AHashSet<&str> = text2.split_whitespace().collect();
    
    if words1.is_empty() && words2.is_empty() {
        return 1.0;
    }
    
    let intersection = words1.intersection(&words2).count();
    let union = words1.union(&words2).count();
    
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

/// Batch compute Jaccard similarity of one text against many.
/// Returns vec of similarities in same order as texts.
#[pyfunction]
pub fn jaccard_similarity_batch(query: &str, texts: Vec<String>) -> Vec<f64> {
    let query_words: AHashSet<&str> = query.split_whitespace().collect();
    
    texts.iter().map(|text| {
        let text_words: AHashSet<&str> = text.split_whitespace().collect();
        
        if query_words.is_empty() && text_words.is_empty() {
            return 1.0;
        }
        
        let intersection = query_words.intersection(&text_words).count();
        let union = query_words.union(&text_words).count();
        
        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }).collect()
}

/// Select item indices using Maximal Marginal Relevance (MMR).
///
/// This is used to reduce near-duplicate results in hybrid retrieval.
///
/// Args:
///     scores: Relevance scores for each item
///     contents: Text content for each item (preferably already lowercased)
///     target_count: Number of items to select
///     lambda_param: Tradeoff between relevance and diversity (0..1)
///
/// Returns:
///     List of selected indices in selection order
#[pyfunction]
#[pyo3(signature = (scores, contents, target_count, lambda_param=0.7))]
pub fn mmr_select_indices(
    py: Python,
    scores: Vec<f32>,
    contents: Vec<String>,
    target_count: usize,
    lambda_param: f32,
) -> PyResult<Vec<usize>> {
    if scores.len() != contents.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "scores and contents must have the same length",
        ));
    }

    let n = scores.len();
    if n == 0 || target_count == 0 {
        return Ok(Vec::new());
    }

    let k = target_count.min(n);
    let lambda = lambda_param.clamp(0.0, 1.0);

    py.detach(|| {
        // Pre-tokenize to avoid repeated splitting.
        let token_sets: Vec<AHashSet<&str>> = contents
            .iter()
            .map(|c| c.split_whitespace().collect())
            .collect();

        let mut selected: Vec<usize> = Vec::with_capacity(k);
        let mut chosen: Vec<bool> = vec![false; n];

        // Always start from the first item (caller should pre-sort by score).
        selected.push(0);
        chosen[0] = true;

        while selected.len() < k {
            let mut best_idx: Option<usize> = None;
            let mut best_score: f32 = f32::NEG_INFINITY;

            for i in 0..n {
                if chosen[i] {
                    continue;
                }

                let mut max_overlap: f32 = 0.0;
                for &j in &selected {
                    let a = &token_sets[i];
                    let b = &token_sets[j];

                    let inter = a.intersection(b).count() as f32;
                    let union = (a.len() + b.len()) as f32 - inter;
                    let overlap = if union <= 0.0 { 0.0 } else { inter / union };
                    if overlap > max_overlap {
                        max_overlap = overlap;
                        if max_overlap >= 1.0 {
                            break;
                        }
                    }
                }

                let mmr = lambda * scores[i] + (1.0 - lambda) * (1.0 - max_overlap);
                if mmr > best_score {
                    best_score = mmr;
                    best_idx = Some(i);
                }
            }

            let idx = match best_idx {
                Some(v) => v,
                None => break,
            };
            selected.push(idx);
            chosen[idx] = true;
        }

        Ok(selected)
    })
}

fn resolve_safe_file(repo_root: &Path, filename: &str) -> PyResult<PathBuf> {
    let rel = Path::new(filename);
    if rel.is_absolute() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "absolute paths are not allowed",
        ));
    }

    // Prevent obvious path traversal before we attempt any filesystem operations.
    // We still canonicalize and enforce repo_root containment below to handle symlinks.
    if rel.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "path traversal is not allowed",
        ));
    }

    let repo_root = repo_root
        .canonicalize()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("invalid repo_path: {e}")))?;

    let candidate = repo_root.join(rel);
    let candidate = candidate
        .canonicalize()
        .map_err(|e| pyo3::exceptions::PyFileNotFoundError::new_err(format!(
            "cannot read file: {filename} ({e})"
        )))?;

    if !candidate.starts_with(&repo_root) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "file escapes repo root",
        ));
    }

    Ok(candidate)
}

/// Extract a line range from a file on disk.
///
/// Returns a dict with:
/// - code
/// - extracted_line_start / extracted_line_end
/// - file_line_count
/// - truncated
#[pyfunction]
#[pyo3(signature = (repo_path, filename, line_start, line_end, max_code_chars=None))]
pub fn extract_code_by_line_range(
    py: Python,
    repo_path: String,
    filename: String,
    line_start: usize,
    line_end: usize,
    max_code_chars: Option<usize>,
) -> PyResult<Py<PyDict>> {
    if line_start < 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "line_start must be >= 1",
        ));
    }
    if line_end < line_start {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "line_end must be >= line_start",
        ));
    }

    let repo_root = Path::new(&repo_path);
    let file_path = resolve_safe_file(repo_root, &filename)?;

    let file = File::open(&file_path)
        .map_err(|e| pyo3::exceptions::PyFileNotFoundError::new_err(format!(
            "cannot read file: {filename} ({e})"
        )))?;
    let mut reader = BufReader::new(file);

    let mut buf = String::new();
    let mut out = String::new();

    let mut line_no: usize = 0;
    let mut extracted_end: isize = line_start as isize - 1;
    let mut truncated: bool = false;
    let mut capture_enabled: bool = true;

    loop {
        buf.clear();
        let bytes = reader
            .read_line(&mut buf)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("read failed: {e}")))?;
        if bytes == 0 {
            break;
        }

        line_no += 1;

        if capture_enabled && line_no >= line_start && line_no <= line_end {
            if let Some(limit) = max_code_chars {
                if out.len() + buf.len() > limit {
                    if out.is_empty() {
                        let remaining = limit.saturating_sub(out.len());
                        out.push_str(&buf.chars().take(remaining).collect::<String>());
                        extracted_end = line_no as isize;
                    }
                    truncated = true;
                    capture_enabled = false;
                } else {
                    out.push_str(&buf);
                    extracted_end = line_no as isize;
                }
            } else {
                out.push_str(&buf);
                extracted_end = line_no as isize;
            }
        }
    }

    let file_line_count = line_no;
    let effective_line_count = std::cmp::max(file_line_count, 1);

    if line_start > effective_line_count {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "line_start beyond end of file ({file_line_count} lines)"
        )));
    }

    let clamped_end = std::cmp::min(line_end, effective_line_count);
    if clamped_end < line_end {
        truncated = true;
    }

    let dict = PyDict::new(py);
    dict.set_item("code", out)?;
    dict.set_item("extracted_line_start", line_start)?;
    dict.set_item(
        "extracted_line_end",
        if extracted_end < 0 { 0 } else { extracted_end as usize },
    )?;
    dict.set_item("file_line_count", file_line_count)?;
    dict.set_item("truncated", truncated)?;

    Ok(dict.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_file_hash() {
        let h1 = compute_file_hash("def foo(): pass");
        let h2 = compute_file_hash("def foo(): pass");
        let h3 = compute_file_hash("def bar(): pass");
        
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
        assert_eq!(h1.len(), 16);
    }

    #[test]
    fn test_jaccard_similarity() {
        // Identical
        assert!((jaccard_similarity("a b c", "a b c") - 1.0).abs() < 0.001);
        
        // 50% overlap
        let sim = jaccard_similarity("a b", "b c");
        assert!((sim - 1.0/3.0).abs() < 0.001); // intersection=1, union=3
        
        // No overlap
        assert!((jaccard_similarity("a b", "c d") - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_jaccard_batch() {
        let results = jaccard_similarity_batch("a b c", vec!["a b c".to_string(), "a b".to_string(), "x y z".to_string()]);
        assert!((results[0] - 1.0).abs() < 0.001);
        assert!(results[1] > 0.5);
        assert!((results[2] - 0.0).abs() < 0.001);
    }
}
