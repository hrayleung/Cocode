//! Utility functions for hashing and similarity calculations.

use pyo3::prelude::*;
use sha2::{Sha256, Digest};
use std::collections::HashSet;

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
    let words1: HashSet<&str> = text1.split_whitespace().collect();
    let words2: HashSet<&str> = text2.split_whitespace().collect();
    
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
    let query_words: HashSet<&str> = query.split_whitespace().collect();
    
    texts.iter().map(|text| {
        let text_words: HashSet<&str> = text.split_whitespace().collect();
        
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
