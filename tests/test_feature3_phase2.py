"""Tests for Feature 3 Phase 2: Substructure Analysis."""

import pytest
import torch
import numpy as np
from feature3.analysis.substructure_mapper import SubstructureMapper
from feature3.analysis.importance_aggregator import ImportanceAggregator


BENZENE = 'c1ccccc1'
NITROBENZENE = 'O=[N+]([O-])c1ccccc1'
ANILINE = 'Nc1ccccc1'
ASPIRIN = 'CC(=O)Oc1ccccc1C(=O)O'
ETHANOL = 'CCO'
INVALID = 'NOT_VALID_SMILES'


class TestSubstructureMapperInit:

    def test_default_patterns_compiled(self):
        m = SubstructureMapper()
        assert len(m.compiled) > 0
        assert 'nitro' in m.compiled
        assert 'benzene' in m.compiled

    def test_custom_patterns(self):
        m = SubstructureMapper(patterns={'methyl': '[CH3]'})
        assert 'methyl' in m.compiled

    def test_invalid_smarts_skipped(self):
        patterns = {'valid': 'c1ccccc1', 'bad': 'NOT###SMARTS'}
        m = SubstructureMapper(patterns=patterns)
        assert 'valid' in m.compiled
        assert 'bad' not in m.compiled

    def test_aggregation_default(self):
        m = SubstructureMapper()
        assert m.aggregation == 'mean'


class TestMapToSubstructures:

    @pytest.fixture
    def mapper(self):
        return SubstructureMapper()

    def test_benzene_detected(self, mapper):
        imp = torch.ones(6) * 0.8
        res = mapper.map_to_substructures(BENZENE, imp)
        assert res['benzene']['present']
        assert abs(res['benzene']['score'] - 0.8) < 0.01

    def test_nitro_detected_in_nitrobenzene(self, mapper):
        imp = torch.ones(9) * 0.7
        res = mapper.map_to_substructures(NITROBENZENE, imp)
        assert res['nitro']['present']
        assert res['nitro']['score'] > 0

    def test_absent_group(self, mapper):
        imp = torch.ones(6) * 0.5
        res = mapper.map_to_substructures(BENZENE, imp)
        assert not res['nitro']['present']
        assert res['nitro']['score'] == 0.0
        assert res['nitro']['frequency'] == 0

    def test_invalid_smiles_returns_empty(self, mapper):
        imp = torch.ones(5) * 0.5
        res = mapper.map_to_substructures(INVALID, imp)
        assert res == {}

    def test_importance_shorter_than_atoms(self, mapper):
        imp = torch.ones(3) * 0.9  # 3 values, 6 atoms in benzene
        res = mapper.map_to_substructures(BENZENE, imp)
        assert isinstance(res, dict)

    def test_importance_longer_than_atoms(self, mapper):
        imp = torch.ones(20) * 0.6  # 20 values, 6 atoms in benzene
        res = mapper.map_to_substructures(BENZENE, imp)
        assert res['benzene']['present']

    def test_max_aggregation(self):
        mapper = SubstructureMapper(aggregation='max')
        imp = torch.tensor([0.1, 0.9, 0.1, 0.1, 0.1, 0.1])
        res = mapper.map_to_substructures(BENZENE, imp)
        if res['benzene']['present']:
            assert res['benzene']['score'] == pytest.approx(0.9, abs=0.05)

    def test_sum_aggregation(self):
        mapper = SubstructureMapper(aggregation='sum')
        imp = torch.ones(6) * 0.5
        res = mapper.map_to_substructures(BENZENE, imp)
        if res['benzene']['present']:
            assert res['benzene']['score'] > 0.5  # Sum > mean

    def test_high_importance_on_toxic_group(self, mapper):
        """Nitro group atoms should get high score if they are important."""
        # Nitrobenzene: atoms 0,1 = N,O,O (nitro), rest = benzene
        # Set high importance on first 3 atoms (nitro group)
        imp = torch.tensor([0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        res = mapper.map_to_substructures(NITROBENZENE, imp)
        # Nitro group should have higher score than benzene
        if res['nitro']['present'] and res['benzene']['present']:
            assert res['nitro']['score'] >= res['benzene']['score']

    def test_frequency_counts(self, mapper):
        imp = torch.ones(13) * 0.5
        res = mapper.map_to_substructures(ASPIRIN, imp)
        if res['benzene']['present']:
            assert res['benzene']['frequency'] >= 1


class TestRankSubstructures:

    @pytest.fixture
    def mapper(self):
        return SubstructureMapper()

    def test_sorted_descending(self, mapper):
        imp = torch.ones(9) * 0.7
        res = mapper.map_to_substructures(NITROBENZENE, imp)
        ranked = mapper.rank_substructures(res, top_k=5)
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limit(self, mapper):
        imp = torch.ones(6) * 0.5
        res = mapper.map_to_substructures(BENZENE, imp)
        ranked = mapper.rank_substructures(res, top_k=3)
        assert len(ranked) <= 3

    def test_present_only_true(self, mapper):
        imp = torch.ones(6) * 0.5
        res = mapper.map_to_substructures(BENZENE, imp)
        ranked = mapper.rank_substructures(res, present_only=True)
        for name, _ in ranked:
            assert res[name]['present']

    def test_returns_list_of_tuples(self, mapper):
        imp = torch.ones(6) * 0.5
        res = mapper.map_to_substructures(BENZENE, imp)
        ranked = mapper.rank_substructures(res)
        for item in ranked:
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], float)


class TestGetAtomGroups:

    @pytest.fixture
    def mapper(self):
        return SubstructureMapper()

    def test_returns_dict(self, mapper):
        result = mapper.get_atom_groups(BENZENE)
        assert isinstance(result, dict)
        assert len(result) == 6  # 6 atoms

    def test_each_value_is_list(self, mapper):
        result = mapper.get_atom_groups(BENZENE)
        for _, groups in result.items():
            assert isinstance(groups, list)

    def test_invalid_smiles(self, mapper):
        result = mapper.get_atom_groups(INVALID)
        assert result == {}


class TestDatasetSummary:

    @pytest.fixture
    def mapper(self):
        return SubstructureMapper()

    def test_basic(self, mapper):
        smiles_list = [BENZENE, NITROBENZENE]
        exps = [
            {'node_importance': torch.ones(6) * 0.7},
            {'node_importance': torch.ones(9) * 0.8},
        ]
        summary = mapper.dataset_summary(smiles_list, exps)
        assert 'benzene' in summary
        assert summary['benzene']['count'] >= 2

    def test_frequency_range(self, mapper):
        smiles_list = [BENZENE, ETHANOL]
        exps = [
            {'node_importance': torch.ones(6) * 0.5},
            {'node_importance': torch.ones(3) * 0.5},
        ]
        summary = mapper.dataset_summary(smiles_list, exps)
        for _, v in summary.items():
            assert 0.0 <= v['frequency'] <= 1.0

    def test_absent_group_has_zero_count(self, mapper):
        smiles_list = [ETHANOL]
        exps = [{'node_importance': torch.ones(3) * 0.5}]
        summary = mapper.dataset_summary(smiles_list, exps)
        if 'nitro' in summary:
            assert summary['nitro']['count'] == 0


class TestImportanceAggregator:

    @pytest.fixture
    def agg(self):
        return ImportanceAggregator()

    def make_exp(self, n_atoms=5, n_feat=10):
        return {
            'edge_mask': torch.rand(8),
            'node_feat_mask': torch.rand(n_feat),
            'node_importance': torch.rand(n_atoms),
            'prediction': float(torch.rand(1)),
        }

    def test_add_and_retrieve(self, agg):
        exps = [self.make_exp() for _ in range(3)]
        agg.add_task_explanations(0, exps)
        feat = agg.get_feature_importance(0)
        assert 'mean' in feat
        assert feat['mean'].shape == (10,)

    def test_empty_task(self, agg):
        feat = agg.get_feature_importance(99)
        assert feat == {}

    def test_sparsity(self, agg):
        exps = [self.make_exp() for _ in range(3)]
        agg.add_task_explanations(0, exps)
        sp = agg.get_average_edge_sparsity(0)
        assert 0.0 <= sp <= 1.0

    def test_prediction_distribution(self, agg):
        exps = [self.make_exp() for _ in range(5)]
        agg.add_task_explanations(0, exps)
        dist = agg.get_prediction_distribution(0)
        assert 'mean' in dist
        assert 'fraction_positive' in dist
        assert dist['count'] == 5

    def test_summary(self, agg):
        exps = [self.make_exp() for _ in range(3)]
        agg.add_task_explanations(0, exps)
        s = agg.summary()
        assert 0 in s
        assert s[0]['n_molecules'] == 3