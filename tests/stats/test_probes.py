from shared_circuits.stats import evaluate_probe_transfer, train_probe


class TestTrainProbe:
    def test_returns_metrics(self, random_activations):
        pos, neg = random_activations
        result = train_probe(pos, neg)
        assert 'auroc' in result
        assert 'accuracy' in result
        assert 'coefficients' in result
        assert 'intercept' in result

    def test_auroc_above_chance(self, random_activations):
        pos, neg = random_activations
        result = train_probe(pos, neg)
        assert result['auroc'] > 0.5

    def test_coefficients_shape(self, random_activations):
        pos, neg = random_activations
        result = train_probe(pos, neg)
        coefs = result['coefficients']
        assert hasattr(coefs, 'shape')
        assert coefs.shape == (pos.shape[1],)


class TestEvaluateProbeTransfer:
    def test_returns_all_metrics(self, random_activations):
        pos, neg = random_activations
        result = evaluate_probe_transfer(pos, neg, pos, neg)
        assert 'train_auroc' in result
        assert 'test_auroc' in result
        assert 'train_accuracy' in result
        assert 'test_accuracy' in result

    def test_train_and_test_above_chance(self, random_activations):
        pos, neg = random_activations
        result = evaluate_probe_transfer(pos, neg, pos, neg)
        assert result['train_auroc'] > 0.5
        assert result['test_auroc'] > 0.5
