use ahash::AHashSet;
use pyo3::prelude::*;
use tree_sitter::{Language, Node, Parser};

fn ts_language(language: &str) -> Option<Language> {
    match language {
        "python" => Some(tree_sitter_python::LANGUAGE.into()),
        "go" => Some(tree_sitter_go::LANGUAGE.into()),
        "rust" => Some(tree_sitter_rust::LANGUAGE.into()),
        "c" => Some(tree_sitter_c::LANGUAGE.into()),
        "cpp" => Some(tree_sitter_cpp::LANGUAGE.into()),
        "javascript" => Some(tree_sitter_javascript::LANGUAGE.into()),
        "typescript" => Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
        "tsx" => Some(tree_sitter_typescript::LANGUAGE_TSX.into()),
        _ => None,
    }
}

fn node_text<'a>(node: Node<'a>, source: &'a [u8]) -> &'a str {
    let start = node.start_byte();
    let end = node.end_byte();
    std::str::from_utf8(&source[start..end]).unwrap_or("")
}

fn walk<'a, F: FnMut(Node<'a>)>(node: Node<'a>, f: &mut F) {
    f(node);
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        walk(child, f);
    }
}

fn push_unique(value: String, seen: &mut AHashSet<String>, out: &mut Vec<String>) {
    if value.is_empty() {
        return;
    }
    if seen.insert(value.clone()) {
        out.push(value);
    }
}

fn strip_quotes(s: &str) -> String {
    let t = s.trim();
    t.trim_matches(['"', '\''].as_ref()).to_string()
}
fn strip_python_string_quotes(s: &str) -> String {
    let t = s.trim();

    // Tree-sitter Python string nodes can include an optional prefix like r/u/f/b (and combinations like fr/rf).
    // Strip it if present so docstrings don't keep the prefix.
    let t = if let Some(idx) = t.find('"').or_else(|| t.find('\'')) {
        let (prefix, rest) = t.split_at(idx);
        if !prefix.is_empty() && prefix.chars().all(|c| c.is_ascii_alphabetic()) {
            rest
        } else {
            t
        }
    } else {
        t
    };

    // Handle triple-quoted strings first.
    for q in ["\"\"\"", "'''"] .iter() {
        if t.starts_with(q) && t.ends_with(q) && t.len() >= q.len() * 2 {
            return t[q.len()..t.len() - q.len()].to_string();
        }
    }

    strip_quotes(t)
}

fn is_test_file(filename: &str) -> bool {
    // Normalize path separators so Windows paths are treated consistently.
    let normalized = filename.replace('\\', "/");
    let lower = normalized.to_lowercase();
    let stem = std::path::Path::new(&normalized)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();

    (stem.contains("test_")
        || stem.contains("_test")
        || stem.contains("tests")
        || stem.contains("spec"))
        || lower.contains("/test")
}

fn is_test_symbol(name: &str, language: &str) -> bool {
    let lower = name.to_lowercase();
    match language {
        "python" => lower.starts_with("test"),
        "go" => name.starts_with("Test") || name.starts_with("Benchmark"),
        "typescript" | "javascript" | "tsx" => {
            lower == "it" || lower == "describe" || lower == "test" || lower.starts_with("test")
        }
        _ => lower.starts_with("test"),
    }
}

fn python_visibility(name: &str) -> String {
    if name.starts_with("__") && !name.ends_with("__") {
        return "private".to_string();
    }
    if name.starts_with('_') {
        return "internal".to_string();
    }
    "public".to_string()
}

fn go_visibility(name: &str) -> String {
    match name.chars().next() {
        Some(c) if c.is_lowercase() => "internal".to_string(),
        _ => "public".to_string(),
    }
}

fn rust_visibility(node: Node, source: &[u8]) -> String {
    // In tree-sitter-rust, public items typically have a visibility_modifier.
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "visibility_modifier" {
            return "public".to_string();
        }
        // Some grammar versions expose "pub" as a token.
        let text = node_text(child, source).trim();
        if text == "pub" {
            return "public".to_string();
        }
    }
    "private".to_string()
}

fn signature_first_line(node: Node, source: &[u8], language: &str) -> String {
    let text = node_text(node, source);
    let mut first = text.lines().next().unwrap_or("").trim().to_string();

    if language == "python" {
        if let Some(idx) = first.rfind(':') {
            first = first[..idx + 1].to_string();
        }
    } else if let Some(idx) = first.find('{') {
        first = first[..idx].trim().to_string();
    }

    if first.len() > 200 {
        first.truncate(200);
    }
    first
}

fn python_docstring(node: Node, source: &[u8]) -> Option<String> {
    let body = node.child_by_field_name("body")?;
    let mut cursor = body.walk();
    let mut iter = body.children(&mut cursor);
    let first = iter.next()?;
    if first.kind() != "expression_statement" {
        return None;
    }
    let mut c = first.walk();
    let expr = first.children(&mut c).next()?;
    if expr.kind() != "string" {
        return None;
    }
    Some(strip_python_string_quotes(node_text(expr, source)))
}

fn leading_comment_docstring(node: Node, source: &[u8]) -> Option<String> {
    let mut parts: Vec<String> = Vec::new();
    let mut prev = node.prev_sibling();

    while let Some(p) = prev {
        match p.kind() {
            "comment" | "line_comment" | "block_comment" => {
                let raw = node_text(p, source).trim().to_string();
                if raw.starts_with("///") {
                    parts.insert(0, raw.trim_start_matches('/').trim().to_string());
                } else if raw.starts_with("//!") {
                    parts.insert(0, raw.trim_start_matches('/').trim().to_string());
                } else if raw.starts_with('#') && !raw.starts_with("#!") {
                    parts.insert(0, raw.trim_start_matches('#').trim().to_string());
                } else if raw.starts_with("/*") {
                    parts.insert(0, raw.trim_matches(['/', '*', ' ', '\n'].as_ref()).to_string());
                }
                prev = p.prev_sibling();
            }
            _ => break,
        }
    }

    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n"))
    }
}

#[pyfunction]
pub fn is_language_supported(language: &str) -> bool {
    ts_language(language).is_some()
}

#[pyfunction]
pub fn extract_imports_ast(code: &str, language: &str) -> Vec<String> {
    if code.trim().is_empty() {
        return Vec::new();
    }

    let lang = match ts_language(language) {
        Some(l) => l,
        None => return Vec::new(),
    };

    let mut parser = Parser::new();
    if parser.set_language(&lang).is_err() {
        return Vec::new();
    }

    let tree = match parser.parse(code, None) {
        Some(t) => t,
        None => return Vec::new(),
    };

    let source = code.as_bytes();
    let root = tree.root_node();

    let mut out: Vec<String> = Vec::new();
    let mut seen: AHashSet<String> = AHashSet::new();

    let mut visit = |node: Node<'_>| {
        match language {
            "python" => match node.kind() {
                "import_statement" => {
                    let mut cursor = node.walk();
                    for child in node.children(&mut cursor) {
                        if child.kind() == "dotted_name" {
                            push_unique(node_text(child, source).trim().to_string(), &mut seen, &mut out);
                        } else if child.kind() == "aliased_import" {
                            if let Some(name) = child.child_by_field_name("name") {
                                push_unique(node_text(name, source).trim().to_string(), &mut seen, &mut out);
                            }
                        }
                    }
                }
                "import_from_statement" => {
                    if let Some(m) = node.child_by_field_name("module_name") {
                        push_unique(node_text(m, source).trim().to_string(), &mut seen, &mut out);
                    }
                }
                _ => {}
            },
            "go" => {
                if node.kind() == "import_spec" {
                    let mut cursor = node.walk();
                    for child in node.children(&mut cursor) {
                        if child.kind() == "interpreted_string_literal" {
                            push_unique(strip_quotes(node_text(child, source)), &mut seen, &mut out);
                        }
                    }
                }
            }
            "rust" => match node.kind() {
                "use_declaration" => {
                    let raw = node_text(node, source);
                    let raw = raw.trim().trim_end_matches(';');
                    let raw = raw.strip_prefix("use").unwrap_or(raw).trim();
                    let path = raw.split_whitespace().next().unwrap_or("");
                    let path = path.split('{').next().unwrap_or(path).trim();
                    let base = path.split("::").next().unwrap_or("").trim();
                    if !base.is_empty() && base != "crate" {
                        push_unique(base.to_string(), &mut seen, &mut out);
                    }
                }
                "mod_item" => {
                    if let Some(name) = node.child_by_field_name("name") {
                        push_unique(node_text(name, source).trim().to_string(), &mut seen, &mut out);
                    }
                }
                "extern_crate_declaration" => {
                    if let Some(name) = node.child_by_field_name("name") {
                        push_unique(node_text(name, source).trim().to_string(), &mut seen, &mut out);
                    }
                }
                _ => {}
            },
            "c" | "cpp" => {
                if node.kind() == "preproc_include" {
                    let mut cursor = node.walk();
                    for child in node.children(&mut cursor) {
                        if child.kind() == "system_lib_string" || child.kind() == "string_literal" {
                            let raw = node_text(child, source).trim();
                            let cleaned = raw.trim_matches(['<', '>', '"'].as_ref()).to_string();
                            push_unique(cleaned, &mut seen, &mut out);
                        }
                    }
                }
            }
            "javascript" | "typescript" | "tsx" => match node.kind() {
                "import_statement" | "export_statement" => {
                    if let Some(src) = node.child_by_field_name("source") {
                        if src.kind() == "string" {
                            push_unique(strip_quotes(node_text(src, source)), &mut seen, &mut out);
                        }
                    }
                }
                "call_expression" => {
                    if let Some(func) = node.child_by_field_name("function") {
                        if func.kind() == "identifier" && node_text(func, source).trim() == "require" {
                            if let Some(args) = node.child_by_field_name("arguments") {
                                let mut cursor = args.walk();
                                for child in args.children(&mut cursor) {
                                    if child.kind() == "string" {
                                        push_unique(strip_quotes(node_text(child, source)), &mut seen, &mut out);
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
                _ => {}
            },
            _ => {}
        }
    };
    walk(root, &mut visit);

    out
}

#[pyfunction]
#[pyo3(signature = (code, language, filename))]
pub fn extract_symbols(
    code: &str,
    language: &str,
    filename: &str,
) -> PyResult<
    Vec<(
        String,
        String,
        usize,
        usize,
        String,
        Option<String>,
        Option<String>,
        String,
        String,
    )>,
> {
    if code.trim().is_empty() {
        return Ok(Vec::new());
    }

    let mut effective_lang = language.to_string();
    if language == "typescript" && filename.to_lowercase().ends_with(".tsx") {
        effective_lang = "tsx".to_string();
    }

    let lang = match ts_language(&effective_lang) {
        Some(l) => l,
        None => return Ok(Vec::new()),
    };

    let mut parser = Parser::new();
    if parser.set_language(&lang).is_err() {
        return Ok(Vec::new());
    }

    let tree = match parser.parse(code, None) {
        Some(t) => t,
        None => return Ok(Vec::new()),
    };

    let source = code.as_bytes();
    let root = tree.root_node();

    let file_is_test = is_test_file(filename);

    let mut out: Vec<(
        String,
        String,
        usize,
        usize,
        String,
        Option<String>,
        Option<String>,
        String,
        String,
    )> = Vec::new();

    fn visit(
        node: Node,
        source: &[u8],
        language: &str,
        filename: &str,
        file_is_test: bool,
        parent_name: Option<String>,
        out: &mut Vec<(
            String,
            String,
            usize,
            usize,
            String,
            Option<String>,
            Option<String>,
            String,
            String,
        )>,
    ) {
        let kind = node.kind();

        // Rust impl blocks: visit children with parent context (type name).
        if language == "rust" && kind == "impl_item" {
            let impl_type = node
                .child_by_field_name("type")
                .map(|n| node_text(n, source).trim().to_string());

            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                visit(
                    child,
                    source,
                    language,
                    filename,
                    file_is_test,
                    impl_type.clone().or_else(|| parent_name.clone()),
                    out,
                );
            }
            return;
        }

        // Handle TypeScript/JavaScript lexical declarations for components/hooks.
        if matches!(language, "typescript" | "tsx" | "javascript") && kind == "lexical_declaration" {
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                if child.kind() != "variable_declarator" {
                    continue;
                }

                let name_node = child.child_by_field_name("name");
                let value_node = child.child_by_field_name("value");
                let Some(name_node) = name_node else { continue };

                let name = node_text(name_node, source).trim().to_string();
                if name.starts_with('[') || name.starts_with('{') {
                    continue;
                }

                let is_fn_value = value_node
                    .map(|v| v.kind() == "arrow_function" || v.kind() == "function")
                    .unwrap_or(false);

                if !is_fn_value {
                    continue;
                }

                let decl_type = if name == "use"
                    || (name.starts_with("use")
                        && name.len() > 3
                        && name.chars().nth(3).unwrap_or('_').is_uppercase())
                {
                    "hook".to_string()
                } else if name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                    "component".to_string()
                } else {
                    "function".to_string()
                };

                let line_start = node.start_position().row as usize + 1;
                let line_end = node.end_position().row as usize + 1;
                let signature = signature_first_line(node, source, language);
                let docstring = leading_comment_docstring(node, source);
                let visibility = "public".to_string();
                let cat = if file_is_test || is_test_symbol(&name, language) {
                    "test".to_string()
                } else {
                    "implementation".to_string()
                };

                out.push((
                    name,
                    decl_type,
                    line_start,
                    line_end,
                    signature,
                    docstring,
                    parent_name.clone(),
                    visibility,
                    cat,
                ));
            }

            return;
        }

        // Determine whether this node defines a symbol and extract name/type.
        let mut symbol_name: Option<String> = None;
        let mut symbol_type: Option<String> = None;
        let mut symbol_parent = parent_name.clone();

        match (language, kind) {
            ("python", "function_definition") => {
                if let Some(name) = node.child_by_field_name("name") {
                    symbol_name = Some(node_text(name, source).trim().to_string());
                    symbol_type = Some("function".to_string());
                }
            }
            ("python", "class_definition") => {
                if let Some(name) = node.child_by_field_name("name") {
                    symbol_name = Some(node_text(name, source).trim().to_string());
                    symbol_type = Some("class".to_string());
                }
            }
            ("go", "function_declaration") => {
                if let Some(name) = node.child_by_field_name("name") {
                    symbol_name = Some(node_text(name, source).trim().to_string());
                    symbol_type = Some("function".to_string());
                }
            }
            ("go", "method_declaration") => {
                if let Some(name) = node.child_by_field_name("name") {
                    symbol_name = Some(node_text(name, source).trim().to_string());
                    symbol_type = Some("method".to_string());
                }
            }
            ("go", "type_declaration") => {
                // Go type_declaration does not expose a direct name field; find type_spec.
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    if child.kind() != "type_spec" {
                        continue;
                    }
                    if let Some(name) = child.child_by_field_name("name") {
                        symbol_name = Some(node_text(name, source).trim().to_string());
                        let mut ty = "class".to_string();
                        if let Some(tn) = child.child_by_field_name("type") {
                            if tn.kind() == "interface_type" {
                                ty = "interface".to_string();
                            }
                        }
                        symbol_type = Some(ty);
                    }
                    break;
                }
            }
            ("rust", "function_item") => {
                if let Some(name) = node.child_by_field_name("name") {
                    symbol_name = Some(node_text(name, source).trim().to_string());
                    symbol_type = Some("function".to_string());
                }
            }
            ("rust", "struct_item") | ("rust", "enum_item") => {
                if let Some(name) = node.child_by_field_name("name") {
                    symbol_name = Some(node_text(name, source).trim().to_string());
                    symbol_type = Some("class".to_string());
                }
            }
            ("rust", "trait_item") => {
                if let Some(name) = node.child_by_field_name("name") {
                    symbol_name = Some(node_text(name, source).trim().to_string());
                    symbol_type = Some("interface".to_string());
                }
            }
            ("typescript" | "tsx", "function_declaration") => {
                if let Some(name) = node.child_by_field_name("name") {
                    symbol_name = Some(node_text(name, source).trim().to_string());
                    symbol_type = Some("function".to_string());
                }
            }
            ("typescript" | "tsx", "class_declaration") => {
                if let Some(name) = node.child_by_field_name("name") {
                    symbol_name = Some(node_text(name, source).trim().to_string());
                    symbol_type = Some("class".to_string());
                }
            }
            ("typescript" | "tsx", "method_definition") => {
                if let Some(name) = node.child_by_field_name("name") {
                    symbol_name = Some(node_text(name, source).trim().to_string());
                    symbol_type = Some("method".to_string());
                }
            }
            ("typescript" | "tsx", "interface_declaration") => {
                if let Some(name) = node.child_by_field_name("name") {
                    symbol_name = Some(node_text(name, source).trim().to_string());
                    symbol_type = Some("interface".to_string());
                }
            }
            ("typescript" | "tsx", "type_alias_declaration") => {
                if let Some(name) = node.child_by_field_name("name") {
                    symbol_name = Some(node_text(name, source).trim().to_string());
                    symbol_type = Some("type".to_string());
                }
            }
            ("typescript" | "tsx", "enum_declaration") => {
                if let Some(name) = node.child_by_field_name("name") {
                    symbol_name = Some(node_text(name, source).trim().to_string());
                    symbol_type = Some("enum".to_string());
                }
            }
            ("javascript", "function_declaration") => {
                if let Some(name) = node.child_by_field_name("name") {
                    symbol_name = Some(node_text(name, source).trim().to_string());
                    symbol_type = Some("function".to_string());
                }
            }
            ("javascript", "class_declaration") => {
                if let Some(name) = node.child_by_field_name("name") {
                    symbol_name = Some(node_text(name, source).trim().to_string());
                    symbol_type = Some("class".to_string());
                }
            }
            ("javascript", "method_definition") => {
                if let Some(name) = node.child_by_field_name("name") {
                    symbol_name = Some(node_text(name, source).trim().to_string());
                    symbol_type = Some("method".to_string());
                }
            }
            _ => {}
        }

        // Emit symbol if present.
        if let (Some(name), Some(ty)) = (symbol_name.clone(), symbol_type.clone()) {
            let actual_type = if symbol_parent.is_some() && ty == "function" {
                "method".to_string()
            } else {
                ty.clone()
            };

            let line_start = node.start_position().row as usize + 1;
            let line_end = node.end_position().row as usize + 1;
            let signature = signature_first_line(node, source, language);

            let docstring = if language == "python" {
                python_docstring(node, source).or_else(|| leading_comment_docstring(node, source))
            } else {
                leading_comment_docstring(node, source)
            };

            let visibility = match language {
                "python" => python_visibility(&name),
                "go" => go_visibility(&name),
                "rust" => rust_visibility(node, source),
                _ => "public".to_string(),
            };

            let cat = if file_is_test || is_test_symbol(&name, language) {
                "test".to_string()
            } else {
                "implementation".to_string()
            };

            out.push((
                name.clone(),
                actual_type.clone(),
                line_start,
                line_end,
                signature,
                docstring,
                symbol_parent.clone(),
                visibility,
                cat,
            ));

            // Recurse into class bodies with parent context.
            if (language == "python" && kind == "class_definition")
                || (matches!(language, "typescript" | "tsx" | "javascript") && kind == "class_declaration")
            {
                symbol_parent = Some(name);
            }
        }

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            visit(
                child,
                source,
                language,
                filename,
                file_is_test,
                symbol_parent.clone(),
                out,
            );
        }
    }

    visit(
        root,
        source,
        &effective_lang,
        filename,
        file_is_test,
        None,
        &mut out,
    );

    Ok(out)
}

#[pyfunction]
pub fn extract_relationships(code: &str, language: &str) -> PyResult<Vec<(String, String, String, usize, f32)>> {
    if code.trim().is_empty() {
        return Ok(Vec::new());
    }

    let lang = match ts_language(language) {
        Some(l) => l,
        None => return Ok(Vec::new()),
    };

    let mut parser = Parser::new();
    if parser.set_language(&lang).is_err() {
        return Ok(Vec::new());
    }

    let tree = match parser.parse(code, None) {
        Some(t) => t,
        None => return Ok(Vec::new()),
    };

    let source = code.as_bytes();
    let root = tree.root_node();

    let mut out: Vec<(String, String, String, usize, f32)> = Vec::new();

    fn visit(node: Node, source: &[u8], language: &str, class_name: Option<String>, out: &mut Vec<(String, String, String, usize, f32)>) {
        let mut current_class = class_name;

        if matches!(node.kind(), "class_definition" | "class_declaration") {
            if let Some(name_node) = node.child_by_field_name("name") {
                current_class = Some(node_text(name_node, source).trim().to_string());

                if language == "python" {
                    // Python: base classes are in argument_list.
                    let mut cursor = node.walk();
                    for child in node.children(&mut cursor) {
                        if child.kind() == "argument_list" {
                            let mut c2 = child.walk();
                            for arg in child.children(&mut c2) {
                                if arg.kind() == "identifier" {
                                    let base = node_text(arg, source).trim().to_string();
                                    if base != "object" && base != "ABC" {
                                        out.push((
                                            current_class.clone().unwrap_or_default(),
                                            base,
                                            "extends".to_string(),
                                            node.start_position().row as usize + 1,
                                            1.0,
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if language == "typescript" || language == "tsx" {
            if let Some(cls) = current_class.clone() {
                match node.kind() {
                    "extends_clause" => {
                        let mut cursor = node.walk();
                        for child in node.children(&mut cursor) {
                            if child.kind() == "type_identifier" || child.kind() == "identifier" {
                                out.push((
                                    cls.clone(),
                                    node_text(child, source).trim().to_string(),
                                    "extends".to_string(),
                                    node.start_position().row as usize + 1,
                                    1.0,
                                ));
                            }
                        }
                    }
                    "implements_clause" => {
                        let mut cursor = node.walk();
                        for child in node.children(&mut cursor) {
                            if child.kind() == "type_identifier" || child.kind() == "identifier" {
                                out.push((
                                    cls.clone(),
                                    node_text(child, source).trim().to_string(),
                                    "implements".to_string(),
                                    node.start_position().row as usize + 1,
                                    1.0,
                                ));
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            visit(child, source, language, current_class.clone(), out);
        }
    }

    visit(root, source, language, None, &mut out);

    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (code, language, current_function_name=None))]
pub fn extract_calls(
    code: &str,
    language: &str,
    current_function_name: Option<String>,
) -> PyResult<Vec<(String, usize, String, Option<String>, Option<String>, bool)>> {
    if code.trim().is_empty() {
        return Ok(Vec::new());
    }

    let lang = match ts_language(language) {
        Some(l) => l,
        None => return Ok(Vec::new()),
    };

    let mut parser = Parser::new();
    if parser.set_language(&lang).is_err() {
        return Ok(Vec::new());
    }

    let tree = match parser.parse(code, None) {
        Some(t) => t,
        None => return Ok(Vec::new()),
    };

    let source = code.as_bytes();
    let root = tree.root_node();

    fn call_context(node: Node) -> Option<String> {
        let mut ctx: Vec<&'static str> = Vec::new();
        let mut parent = node.parent();
        while let Some(p) = parent {
            match p.kind() {
                "for_statement" | "while_statement" | "for_in_clause" => ctx.push("loop"),
                "if_statement" | "elif_clause" | "else_clause" | "conditional_expression" => ctx.push("conditional"),
                "try_statement" | "except_clause" => ctx.push("try_block"),
                "with_statement" => ctx.push("with_block"),
                "lambda" => ctx.push("lambda"),
                _ => {}
            }
            parent = p.parent();
        }
        if ctx.is_empty() {
            None
        } else {
            Some(ctx.join(", "))
        }
    }

    let mut out: Vec<(String, usize, String, Option<String>, Option<String>, bool)> = Vec::new();

    let mut visit = |node: Node<'_>| {
        match language {
            "python" => {
                if node.kind() != "call" {
                    return;
                }

                let Some(func_node) = node.child_by_field_name("function") else { return };

                let mut function_name: Option<String> = None;
                let mut object_name: Option<String> = None;
                let mut call_type = "function_call".to_string();

                if func_node.kind() == "identifier" {
                    function_name = Some(node_text(func_node, source).trim().to_string());
                } else if func_node.kind() == "attribute" {
                    let object_node = func_node.child_by_field_name("object");
                    let attr_node = func_node.child_by_field_name("attribute");

                    if let Some(attr) = attr_node {
                        function_name = Some(node_text(attr, source).trim().to_string());
                        call_type = "method_call".to_string();

                        if let Some(obj) = object_node {
                            if obj.kind() == "identifier" {
                                object_name = Some(node_text(obj, source).trim().to_string());
                            } else if obj.kind() == "call" {
                                object_name = Some("[chained]".to_string());
                            }
                        }
                    }
                }

                let Some(fname) = function_name else { return };

                let line_number = node.start_position().row as usize + 1;
                let ctx = call_context(node);
                let is_recursive = current_function_name
                    .as_ref()
                    .map(|n| n == &fname)
                    .unwrap_or(false);

                out.push((fname, line_number, call_type, ctx, object_name, is_recursive));
            }
            "go" => {
                if node.kind() != "call_expression" {
                    return;
                }

                let Some(func_node) = node.child_by_field_name("function") else { return };

                let mut function_name: Option<String> = None;
                let mut object_name: Option<String> = None;
                let mut call_type = "function_call".to_string();

                if func_node.kind() == "identifier" {
                    function_name = Some(node_text(func_node, source).trim().to_string());
                } else if func_node.kind() == "selector_expression" {
                    let operand = func_node.child_by_field_name("operand");
                    let field = func_node.child_by_field_name("field");
                    if let Some(field) = field {
                        function_name = Some(node_text(field, source).trim().to_string());
                        call_type = "method_call".to_string();
                        if let Some(op) = operand {
                            if op.kind() == "identifier" {
                                object_name = Some(node_text(op, source).trim().to_string());
                            }
                        }
                    }
                }

                let Some(fname) = function_name else { return };
                let line_number = node.start_position().row as usize + 1;

                out.push((fname, line_number, call_type, None, object_name, false));
            }
            _ => {}
        }
    };
    walk(root, &mut visit);

    Ok(out)
}
