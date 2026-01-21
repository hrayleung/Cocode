use pyo3::prelude::*;
use rayon::prelude::*;
use ahash::AHashSet;
use std::sync::LazyLock;

static CODE_STOP_WORDS: LazyLock<AHashSet<&'static str>> = LazyLock::new(|| {
    [
        // Language keywords (control flow)
        "if", "else", "elif", "then", "endif", "switch", "case", "default",
        "for", "while", "do", "loop", "break", "continue", "goto", "return",
        "try", "catch", "except", "finally", "throw", "raise", "throws",
        "yield", "await", "async",
        
        // Declarations
        "def", "func", "fn", "function", "proc", "sub", "method",
        "class", "struct", "enum", "interface", "trait", "type", "typedef",
        "var", "let", "const", "val", "mut", "static", "final", "readonly",
        "import", "from", "export", "require", "include", "using", "use",
        "package", "module", "mod", "namespace", "crate", "extern",
        
        // OOP keywords
        "new", "this", "self", "super", "extends", "implements", "override",
        "public", "private", "protected", "internal", "abstract", "virtual",
        "sealed", "partial",
        
        // Type keywords
        "void", "null", "nil", "none", "undefined", "true", "false",
        "int", "integer", "float", "double", "bool", "boolean", "string", "str",
        "char", "byte", "short", "long", "unsigned", "signed", "size",
        "object", "any", "dynamic", "auto",
        
        // Rust specific
        "impl", "pub", "where", "dyn", "ref", "box", "move", "unsafe", "match",
        
        // Common short identifiers (loop vars, temp vars)
        "i", "j", "k", "n", "m", "x", "y", "z", "a", "b", "c", "e", "f", "p", "q", "r", "s", "t", "v", "w",
        "id", "ok", "err", "io", "os", "fs", "db", "ui", "tx", "rx",
        
        // Common generic variable names
        "tmp", "temp", "val", "var", "arg", "args", "param", "params",
        "ret", "res", "result", "out", "output", "in", "input",
        "buf", "buffer", "ptr", "ref", "ctx", "cfg", "opt", "opts",
        "idx", "len", "num", "cnt", "pos", "key", "val",
        
        // Common method prefixes (too generic alone)
        "get", "set", "has", "is", "can", "should", "will", "did", "was", "are", "do",
        "add", "remove", "delete", "update", "create", "init", "load", "save",
        "read", "write", "open", "close", "start", "stop", "run", "exec",
        "on", "to", "from", "with", "by", "at", "of", "as", "or", "and", "not",
        
        // English articles/prepositions
        "a", "an", "the", "be", "it", "its",
        
        // Common but meaningless alone
        "data", "info", "item", "items", "list", "array", "map", "set",
        "node", "elem", "element", "entry", "record", "row", "col",
        "src", "dst", "source", "dest", "target", "origin",
        "old", "new", "prev", "next", "cur", "current", "last", "first",
        "min", "max", "sum", "avg", "total", "count",
        "name", "value", "type", "kind", "mode", "state", "status",
        "msg", "message", "text", "content", "body", "payload",
        "path", "file", "dir", "url", "uri",
        "fn", "cb", "callback", "handler", "listener", "observer",
    ].into_iter().collect()
});

const MIN_TOKEN_LENGTH: usize = 2;

/// Split camelCase/PascalCase into words
fn split_camel_case(text: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    
    for i in 0..chars.len() {
        let c = chars[i];
        if c.is_uppercase() {
            // Check if this starts a new word
            if !current.is_empty() {
                // Handle acronyms: if previous was uppercase and next is lowercase, split before current
                let prev_upper = i > 0 && chars[i - 1].is_uppercase();
                let next_lower = i + 1 < chars.len() && chars[i + 1].is_lowercase();
                
                if !prev_upper || next_lower {
                    result.push(current);
                    current = String::new();
                }
            }
        }
        current.push(c);
    }
    
    if !current.is_empty() {
        result.push(current);
    }
    
    result
}

/// Extract code tokens from text, handling camelCase, snake_case, etc.
#[pyfunction]
pub fn extract_code_tokens(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    
    // Extract identifiers: sequences of alphanumeric + underscore/dash
    let mut current = String::new();
    
    for c in text.chars() {
        if c.is_alphanumeric() || c == '_' || c == '-' {
            current.push(c);
        } else if !current.is_empty() {
            if current.len() >= MIN_TOKEN_LENGTH {
                process_identifier(&current, &mut tokens);
            }
            current.clear();
        }
    }
    
    if current.len() >= MIN_TOKEN_LENGTH {
        process_identifier(&current, &mut tokens);
    }
    
    tokens
}

fn process_identifier(identifier: &str, tokens: &mut Vec<String>) {
    // Split by underscores and dashes
    for segment in identifier.split(|c| c == '_' || c == '-') {
        if segment.is_empty() {
            continue;
        }
        
        // Split camelCase
        for part in split_camel_case(segment) {
            let normalized = part.to_lowercase();
            if normalized.len() >= MIN_TOKEN_LENGTH 
                && !CODE_STOP_WORDS.contains(normalized.as_str()) 
            {
                tokens.push(normalized);
            }
        }
    }
}

/// Tokenize a search query
#[pyfunction]
pub fn tokenize_for_search(query: &str) -> Vec<String> {
    let code_tokens = extract_code_tokens(query);
    
    // Also extract simple words
    let mut all_tokens: AHashSet<String> = code_tokens.into_iter().collect();
    
    let mut current = String::new();
    for c in query.chars() {
        if c.is_alphabetic() {
            current.push(c.to_ascii_lowercase());
        } else if !current.is_empty() {
            if current.len() >= MIN_TOKEN_LENGTH 
                && !CODE_STOP_WORDS.contains(current.as_str()) 
            {
                all_tokens.insert(current.clone());
            }
            current.clear();
        }
    }
    if current.len() >= MIN_TOKEN_LENGTH && !CODE_STOP_WORDS.contains(current.as_str()) {
        all_tokens.insert(current);
    }
    
    all_tokens.into_iter().collect()
}

/// Batch tokenize multiple texts in parallel
#[pyfunction]
pub fn batch_extract_tokens(py: Python, texts: Vec<String>) -> PyResult<Vec<Vec<String>>> {
    py.detach(|| {
        Ok(texts.par_iter()
            .map(|t| extract_code_tokens(t))
            .collect())
    })
}

/// Batch tokenize queries in parallel
#[pyfunction]
pub fn batch_tokenize_queries(py: Python, queries: Vec<String>) -> PyResult<Vec<Vec<String>>> {
    py.detach(|| {
        Ok(queries.par_iter()
            .map(|q| tokenize_for_search(q))
            .collect())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_camel_case() {
        assert_eq!(split_camel_case("getUserName"), vec!["get", "User", "Name"]);
        assert_eq!(split_camel_case("XMLParser"), vec!["XML", "Parser"]);
        assert_eq!(split_camel_case("getHTTPResponse"), vec!["get", "HTTP", "Response"]);
        assert_eq!(split_camel_case("simple"), vec!["simple"]);
    }

    #[test]
    fn test_extract_code_tokens() {
        let tokens = extract_code_tokens("getUserById");
        assert!(tokens.contains(&"user".to_string()));
        assert!(!tokens.contains(&"by".to_string())); // too short or stop word
        assert!(!tokens.contains(&"id".to_string())); // too short
        
        let tokens = extract_code_tokens("user_name_validator");
        assert!(tokens.contains(&"user".to_string()));
        assert!(!tokens.contains(&"name".to_string())); // stop word
        assert!(tokens.contains(&"validator".to_string()));
    }

    #[test]
    fn test_tokenize_for_search() {
        let tokens = tokenize_for_search("find user authentication");
        assert!(tokens.contains(&"find".to_string()));
        assert!(tokens.contains(&"user".to_string()));
        assert!(tokens.contains(&"authentication".to_string()));
    }
}
