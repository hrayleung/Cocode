"""Unit tests for file categorizer."""

import pytest

from src.retrieval.file_categorizer import (
    categorize_file,
    get_category_boost,
    apply_category_boosting,
)


class TestFileCategorizerDetection:
    """Tests for categorize_file function."""

    # Python test files
    @pytest.mark.parametrize("filename", [
        "test_foo.py",
        "test_bar_baz.py",
        "foo_test.py",
        "bar_baz_test.py",
        "src/test_module.py",
        "src/module_test.py",
    ])
    def test_python_test_files(self, filename):
        assert categorize_file(filename) == "test"

    # JavaScript/TypeScript test files
    @pytest.mark.parametrize("filename", [
        "foo.test.ts",
        "bar.test.tsx",
        "baz.test.js",
        "qux.test.jsx",
        "foo.spec.ts",
        "bar.spec.tsx",
        "baz.spec.js",
        "qux.spec.jsx",
        "src/components/Button.test.tsx",
    ])
    def test_js_ts_test_files(self, filename):
        assert categorize_file(filename) == "test"

    # Go test files
    @pytest.mark.parametrize("filename", [
        "foo_test.go",
        "bar_baz_test.go",
        "pkg/handler_test.go",
    ])
    def test_go_test_files(self, filename):
        assert categorize_file(filename) == "test"

    # Rust test files
    @pytest.mark.parametrize("filename", [
        "foo_test.rs",
        "bar_test.rs",
        "src/lib_test.rs",
    ])
    def test_rust_test_files(self, filename):
        assert categorize_file(filename) == "test"

    # Test directories
    @pytest.mark.parametrize("filename", [
        "tests/test_main.py",
        "test/integration.py",
        "__tests__/Button.test.tsx",
        "spec/models/user_spec.rb",
    ])
    def test_test_directories(self, filename):
        assert categorize_file(filename) == "test"

    # Java and C# test files
    @pytest.mark.parametrize("filename", [
        "FooTest.java",
        "BarBazTest.java",
        "FooTest.cs",
        "FooTests.cs",
    ])
    def test_java_csharp_test_files(self, filename):
        assert categorize_file(filename) == "test"

    # Ruby spec files
    def test_ruby_spec_files(self):
        assert categorize_file("user_spec.rb") == "test"
        assert categorize_file("spec/user_spec.rb") == "test"

    # Documentation files
    @pytest.mark.parametrize("filename", [
        "README.md",
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "docs/api.md",
        "doc/guide.mdx",
        "notes.rst",
        "LICENSE",
        "README",
    ])
    def test_documentation_files(self, filename):
        assert categorize_file(filename) == "documentation"

    # Config files
    @pytest.mark.parametrize("filename", [
        "config.yaml",
        "config.yml",
        "pyproject.toml",
        "package.json",
        "tsconfig.json",
        ".eslintrc",
        ".prettierrc",
        ".env",
        ".env.local",
        "Makefile",
        "Dockerfile",
        "docker-compose.yml",
    ])
    def test_config_files(self, filename):
        assert categorize_file(filename) == "config"

    # Implementation files (should NOT match other categories)
    @pytest.mark.parametrize("filename", [
        "main.py",
        "app.ts",
        "handler.go",
        "lib.rs",
        "service.java",
        "controller.rb",
        "index.js",
        "utils.tsx",
        "src/core/manager.py",
        "pkg/api/routes.go",
        "internal/service.rs",
        # Files that contain "test" in name but aren't test files
        "test_utils.py",  # This IS a test file (test_ prefix)
        "attestation.py",  # Contains "test" but not a test file
        "contest.go",      # Contains "test" but not a test file
    ])
    def test_implementation_files(self, filename):
        # Special case: test_utils.py is a test file
        if filename == "test_utils.py":
            assert categorize_file(filename) == "test"
        else:
            assert categorize_file(filename) == "implementation"

    # Edge cases with paths
    def test_nested_paths(self):
        assert categorize_file("src/lib/core/manager.py") == "implementation"
        assert categorize_file("src/tests/test_manager.py") == "test"
        assert categorize_file("docs/api/endpoints.md") == "documentation"

    # Windows-style paths
    def test_windows_paths(self):
        assert categorize_file("src\\tests\\test_manager.py") == "test"
        assert categorize_file("docs\\api\\endpoints.md") == "documentation"


class TestGetCategoryBoost:
    """Tests for get_category_boost function."""

    def test_implementation_boost(self):
        boost = get_category_boost("main.py")
        assert boost == 1.0  # Default implementation weight

    def test_test_boost(self):
        boost = get_category_boost("test_main.py")
        assert boost == 0.3  # Configured test weight

    def test_documentation_boost(self):
        boost = get_category_boost("README.md")
        assert boost == 0.7  # Default documentation weight

    def test_config_boost(self):
        boost = get_category_boost("config.yaml")
        assert boost == 0.6  # Default config weight


class TestApplyCategoryBoosting:
    """Tests for apply_category_boosting function."""

    def test_boosting_reorders_results(self):
        """Test that boosting causes re-ordering based on file category."""

        # Create mock results (need objects with filename and score attrs)
        class MockResult:
            def __init__(self, filename, score):
                self.filename = filename
                self.score = score

        results = [
            MockResult("test_foo.py", 0.9),      # Test file, high score
            MockResult("README.md", 0.85),       # Doc file, medium score
            MockResult("manager.py", 0.8),       # Implementation, lower score
        ]

        # Before boosting, test_foo.py is first
        assert results[0].filename == "test_foo.py"

        # Apply boosting
        apply_category_boosting(results, sort=True)

        # After boosting:
        # - test_foo.py: 0.9 * 0.3 = 0.27
        # - README.md: 0.85 * 0.7 = 0.595
        # - manager.py: 0.8 * 1.0 = 0.8
        # So order should be: manager.py, README.md, test_foo.py
        assert results[0].filename == "manager.py"
        assert results[1].filename == "README.md"
        assert results[2].filename == "test_foo.py"

    def test_boosting_preserves_order_when_no_sort(self):
        """Test that results maintain order when sort=False."""

        class MockResult:
            def __init__(self, filename, score):
                self.filename = filename
                self.score = score

        results = [
            MockResult("test_foo.py", 0.9),
            MockResult("manager.py", 0.8),
        ]

        apply_category_boosting(results, sort=False)

        # Order preserved but scores modified
        assert results[0].filename == "test_foo.py"
        assert results[0].score == pytest.approx(0.27)  # 0.9 * 0.3
        assert results[1].filename == "manager.py"
        assert results[1].score == pytest.approx(0.8)
