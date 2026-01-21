use pyo3::prelude::*;

mod vector_ops;
mod bm25_engine;
mod graph_algos;
mod tokenizer;

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
    m.add_function(wrap_pyfunction!(graph_algos::strongly_connected_components, m)?)?;

    // Tokenizer
    m.add_function(wrap_pyfunction!(tokenizer::extract_code_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(tokenizer::tokenize_for_search, m)?)?;
    m.add_function(wrap_pyfunction!(tokenizer::batch_extract_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(tokenizer::batch_tokenize_queries, m)?)?;

    Ok(())
}
