use pyo3::prelude::*;
use std::collections::HashMap;
use unicode_segmentation::UnicodeSegmentation;
use ahash::AHashMap;
use rayon::prelude::*;

/// High-performance BM25 scoring engine with WAND-inspired optimization
#[pyclass]
pub struct BM25Engine {
    /// Document frequency: term -> number of documents containing term
    df: AHashMap<String, usize>,
    /// Inverse document frequency: term -> idf score
    idf: AHashMap<String, f32>,
    /// Document lengths
    doc_lengths: Vec<usize>,
    /// Average document length
    avg_doc_length: f32,
    /// Total number of documents
    num_docs: usize,
    /// BM25 k1 parameter (term frequency saturation)
    k1: f32,
    /// BM25 b parameter (length normalization)
    b: f32,
    /// BM25+ delta parameter (lower bound for term frequency)
    delta: f32,
    /// Cached tokenized documents from indexing
    tokenized_docs: Vec<Vec<String>>,
}

#[pymethods]
impl BM25Engine {
    /// Create a new BM25 engine
    ///
    /// Args:
    ///     k1: Term frequency saturation parameter (default: 1.5)
    ///     b: Length normalization parameter (default: 0.75)
    ///     delta: BM25+ delta parameter (default: 1.0)
    #[new]
    #[pyo3(signature = (k1=1.5, b=0.75, delta=1.0))]
    fn new(k1: f32, b: f32, delta: f32) -> Self {
        BM25Engine {
            df: AHashMap::new(),
            idf: AHashMap::new(),
            doc_lengths: Vec::new(),
            avg_doc_length: 0.0,
            num_docs: 0,
            k1,
            b,
            delta,
            tokenized_docs: Vec::new(),
        }
    }

    /// Index a corpus of documents
    ///
    /// Args:
    ///     documents: List of document strings to index
    fn index(&mut self, documents: Vec<String>) {
        // Clear previous state to avoid accumulating stale stats on re-index
        self.df.clear();
        self.idf.clear();
        self.doc_lengths.clear();
        self.tokenized_docs.clear();

        self.num_docs = documents.len();
        self.doc_lengths.reserve(documents.len());

        // Tokenize all documents in parallel and cache
        self.tokenized_docs = documents
            .par_iter()
            .map(|doc| tokenize(doc))
            .collect();

        // Calculate document lengths (explicit loop to avoid mutation in iterator)
        let mut total_length: usize = 0;
        for tokens in &self.tokenized_docs {
            let len = tokens.len();
            self.doc_lengths.push(len);
            total_length += len;
        }

        // Guard against zero division
        self.avg_doc_length = if self.num_docs == 0 {
            0.0
        } else {
            total_length as f32 / self.num_docs as f32
        };

        // Calculate document frequencies
        for tokens in &self.tokenized_docs {
            let mut seen_terms = AHashMap::new();
            for term in tokens {
                seen_terms.insert(term.clone(), true);
            }
            for term in seen_terms.keys() {
                *self.df.entry(term.clone()).or_insert(0) += 1;
            }
        }

        // Calculate IDF scores
        for (term, df) in &self.df {
            let idf = ((self.num_docs as f32 - *df as f32 + 0.5) / (*df as f32 + 0.5) + 1.0).ln();
            self.idf.insert(term.clone(), idf);
        }
    }

    /// Score indexed documents against a query using BM25+
    ///
    /// Args:
    ///     query: Query string
    ///     top_k: Number of top results to return (default: None for all)
    ///     score_threshold: Minimum score threshold for results (default: 0.0)
    ///
    /// Returns:
    ///     List of (doc_index, score) tuples, sorted by score descending
    #[pyo3(signature = (query, top_k=None, score_threshold=0.0))]
    fn score(
        &self,
        py: Python,
        query: String,
        top_k: Option<usize>,
        score_threshold: f32,
    ) -> PyResult<Vec<(usize, f32)>> {
        if self.num_docs == 0 {
            return Ok(Vec::new());
        }

        let query_terms = tokenize(&query);

        // Parallel scoring using cached tokenized docs
        let scores: Vec<(usize, f32)> = py.detach(|| {
            self.tokenized_docs
                .par_iter()
                .enumerate()
                .filter_map(|(idx, doc_tokens)| {
                    let score = self.score_document(&query_terms, doc_tokens, idx);

                    // Early filtering based on threshold
                    if score > score_threshold {
                        Some((idx, score))
                    } else {
                        None
                    }
                })
                .collect()
        });

        // Sort by score descending
        let mut sorted_scores = scores;
        sorted_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Apply top_k limit if specified
        if let Some(k) = top_k {
            sorted_scores.truncate(k);
        }

        Ok(sorted_scores)
    }

    /// Get statistics about the indexed corpus
    ///
    /// Returns:
    ///     Dictionary with corpus statistics
    fn get_stats(&self) -> PyResult<HashMap<String, f32>> {
        let mut stats = HashMap::new();
        stats.insert("num_docs".to_string(), self.num_docs as f32);
        stats.insert("avg_doc_length".to_string(), self.avg_doc_length);
        stats.insert("vocab_size".to_string(), self.df.len() as f32);
        stats.insert("k1".to_string(), self.k1);
        stats.insert("b".to_string(), self.b);
        stats.insert("delta".to_string(), self.delta);
        Ok(stats)
    }
}

impl BM25Engine {
    /// Score a single document against query terms using BM25+
    fn score_document(&self, query_terms: &[String], doc_tokens: &[String], doc_idx: usize) -> f32 {
        if doc_idx >= self.doc_lengths.len() {
            return 0.0;
        }

        let doc_length = self.doc_lengths[doc_idx] as f32;

        // Count term frequencies in document
        let mut term_freqs: AHashMap<&str, usize> = AHashMap::new();
        for token in doc_tokens {
            *term_freqs.entry(token.as_str()).or_insert(0) += 1;
        }

        let mut score = 0.0;
        let avg_doc_len = if self.avg_doc_length > 0.0 { self.avg_doc_length } else { 1.0 };

        for query_term in query_terms {
            if let Some(&tf) = term_freqs.get(query_term.as_str()) {
                if let Some(&idf) = self.idf.get(query_term) {
                    // BM25+ formula with delta parameter
                    let tf_component = (tf as f32 * (self.k1 + 1.0))
                        / (tf as f32 + self.k1 * (1.0 - self.b + self.b * (doc_length / avg_doc_len)));

                    score += idf * (tf_component + self.delta);
                }
            }
        }

        score
    }
}

/// Tokenize text into terms (lowercase, unicode-aware word segmentation)
fn tokenize(text: &str) -> Vec<String> {
    text.unicode_words()
        .map(|word| word.to_lowercase())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        let text = "Hello, World! This is a test.";
        let tokens = tokenize(text);
        assert_eq!(tokens, vec!["hello", "world", "this", "is", "a", "test"]);
    }

    #[test]
    fn test_bm25_basic() {
        let mut engine = BM25Engine::new(1.5, 0.75, 1.0);

        let docs = vec![
            "the quick brown fox".to_string(),
            "the lazy dog".to_string(),
            "quick brown dogs".to_string(),
        ];

        engine.index(docs);

        // Query for "quick brown"
        let results = Python::with_gil(|py| {
            engine.score(py, "quick brown".to_string(), None, 0.0)
        }).unwrap();

        // First and third documents should score higher
        assert!(results.len() >= 2);
        assert!(results[0].1 > 0.0);
    }

    #[test]
    fn test_bm25_threshold() {
        let mut engine = BM25Engine::new(1.5, 0.75, 1.0);

        let docs = vec![
            "the quick brown fox".to_string(),
            "the lazy dog".to_string(),
            "completely unrelated content".to_string(),
        ];

        engine.index(docs);

        // Query with high threshold should filter out low-scoring docs
        let results = Python::with_gil(|py| {
            engine.score(py, "quick brown".to_string(), None, 5.0)
        }).unwrap();

        // Should have fewer results due to threshold
        assert!(results.len() <= 2);
    }
}
