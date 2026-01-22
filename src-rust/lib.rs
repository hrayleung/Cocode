use pyo3::prelude::*;

mod vector_ops;
mod bm25_engine;
mod graph_algos;
mod tokenizer;
mod utils;
mod code_parser;

#[pymodule]
fn cocode_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Vector operations
    m.add_function(wrap_pyfunction!(vector_ops::cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(vector_ops::cosine_similarity_batch, m)?)?;
    m.add_function(wrap_pyfunction!(vector_ops::reciprocal_rank_fusion, m)?)?;
    m.add_function(wrap_pyfunction!(vector_ops::reciprocal_rank_fusion_weighted, m)?)?;

    // BM25
    m.add_class::<bm25_engine::BM25Engine>()?;

    // Graph algorithms
    m.add_function(wrap_pyfunction!(graph_algos::pagerank, m)?)?;
    m.add_function(wrap_pyfunction!(graph_algos::bfs_expansion, m)?)?;
    m.add_function(wrap_pyfunction!(graph_algos::bfs_traversal_edges, m)?)?;
    m.add_function(wrap_pyfunction!(graph_algos::strongly_connected_components, m)?)?;

    // Tokenizer
    m.add_function(wrap_pyfunction!(tokenizer::extract_code_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(tokenizer::tokenize_for_search, m)?)?;
    m.add_function(wrap_pyfunction!(tokenizer::batch_extract_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(tokenizer::batch_tokenize_queries, m)?)?;

    // Utils
    m.add_function(wrap_pyfunction!(utils::compute_file_hash, m)?)?;
    m.add_function(wrap_pyfunction!(utils::jaccard_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(utils::jaccard_similarity_batch, m)?)?;
    m.add_function(wrap_pyfunction!(utils::mmr_select_indices, m)?)?;
    m.add_function(wrap_pyfunction!(utils::extract_code_by_line_range, m)?)?;

    // Code parsing (Tree-sitter)
    m.add_function(wrap_pyfunction!(code_parser::is_language_supported, m)?)?;
    m.add_function(wrap_pyfunction!(code_parser::extract_imports_ast, m)?)?;
    m.add_function(wrap_pyfunction!(code_parser::extract_symbols, m)?)?;
    m.add_function(wrap_pyfunction!(code_parser::extract_relationships, m)?)?;
    m.add_function(wrap_pyfunction!(code_parser::extract_calls, m)?)?;

    Ok(())
}
