"""Tests for shader_variants_codegen._collect_variants."""

import pytest

from repo_tools.shader_variants_codegen import _collect_variants


def _config(shaders):
    return {"slangc": {"shaders": shaders}}


class TestCollectVariants:
    def test_empty_config(self):
        assert _collect_variants({}) == []

    def test_empty_shaders(self):
        assert _collect_variants(_config([])) == []

    def test_implicit_base_variant_is_skipped(self):
        # A shader with no variants and no defines is the implicit base
        # variant -- EmbeddedCompiler returns source_key unchanged for empty
        # defines, so it shouldn't appear in the map.
        shaders = [{"input": "a.slang", "output": "a.wgsl"}]
        assert _collect_variants(_config(shaders)) == []

    def test_top_level_defines_map_to_base_key(self):
        # Top-level defines with no explicit variants: the base output
        # (suffix="") is compiled WITH those defines, so EmbeddedCompiler
        # must map `defines=['FOO']` back to the base source_key.
        shaders = [{
            "input": "a.slang",
            "output": "a.wgsl",
            "defines": ["FOO"],
        }]
        assert _collect_variants(_config(shaders)) == [("FOO\n", "")]

    def test_single_variant_with_defines(self):
        shaders = [{
            "input": "forward.slang",
            "output": "forward.wgsl",
            "variants": [
                {},
                {"defines": ["NO_DEBUG_TARGETS"], "suffix": "_no_debug"},
            ],
        }]
        result = _collect_variants(_config(shaders))
        assert result == [("NO_DEBUG_TARGETS\n", "_no_debug")]

    def test_canonical_defines_sorted(self):
        shaders = [{
            "variants": [
                {"defines": ["BETA", "ALPHA"], "suffix": "_x"},
            ],
        }]
        result = _collect_variants(_config(shaders))
        assert result == [("ALPHA\nBETA\n", "_x")]

    def test_duplicate_variant_across_shaders_deduped(self):
        shaders = [
            {
                "variants": [
                    {"defines": ["FOO"], "suffix": "_foo"},
                ],
            },
            {
                "variants": [
                    {"defines": ["FOO"], "suffix": "_foo"},
                ],
            },
        ]
        result = _collect_variants(_config(shaders))
        assert result == [("FOO\n", "_foo")]

    def test_conflicting_suffix_raises(self):
        shaders = [{
            "variants": [
                {"defines": ["FOO"], "suffix": "_foo"},
                {"defines": ["FOO"], "suffix": "_bar"},
            ],
        }]
        with pytest.raises(ValueError, match="conflicting"):
            _collect_variants(_config(shaders))

    def test_non_dict_shader_raises(self):
        with pytest.raises(ValueError, match="Invalid shader entry"):
            _collect_variants(_config(["not a dict"]))

    def test_non_dict_variant_raises(self):
        shaders = [{"variants": ["not a dict"]}]
        with pytest.raises(ValueError, match="Invalid variant entry"):
            _collect_variants(_config(shaders))
