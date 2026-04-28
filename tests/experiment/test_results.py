import json

import pytest

from shared_circuits.experiment import load_results, model_slug, save_results


class TestModelSlug:
    def test_replaces_slashes(self):
        assert model_slug('Qwen/Qwen2.5-1.5B-Instruct') == 'Qwen_Qwen2.5_1.5B_Instruct'

    def test_replaces_dashes(self):
        assert model_slug('gemma-2-2b-it') == 'gemma_2_2b_it'


class TestSaveResults:
    def test_creates_file(self, tmp_path):
        data = {'model': 'test', 'score': 0.95}
        path = save_results(data, 'test_exp', 'gemma-2-2b-it', results_dir=tmp_path)
        assert path.exists()
        assert path.name == 'test_exp_gemma_2_2b_it.json'

    def test_json_content(self, tmp_path):
        data = {'key': 'value', 'number': 42}
        path = save_results(data, 'exp', 'model-name', results_dir=tmp_path)
        loaded = json.loads(path.read_text())
        assert loaded == data

    def test_creates_directory(self, tmp_path):
        out_dir = tmp_path / 'nested' / 'dir'
        save_results({'a': 1}, 'exp', 'model', results_dir=out_dir)
        assert out_dir.exists()

    def test_slug_handles_slashes(self, tmp_path):
        path = save_results({}, 'exp', 'Qwen/Qwen2.5-1.5B-Instruct', results_dir=tmp_path)
        assert 'Qwen_Qwen2.5_1.5B_Instruct' in path.name


class TestLoadResults:
    def test_roundtrip(self, tmp_path):
        data = {'model': 'test', 'metrics': [1, 2, 3]}
        save_results(data, 'roundtrip', 'test-model', results_dir=tmp_path)
        loaded = load_results('roundtrip', 'test-model', results_dir=tmp_path)
        assert loaded == data

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_results('nonexistent', 'fake-model', results_dir=tmp_path)
