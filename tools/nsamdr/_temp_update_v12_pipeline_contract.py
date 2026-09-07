from pathlib import Path

path = Path('tools/nsamdr/tests/test_pass_driven_pipeline_contract.py')
text = path.read_text(encoding='utf-8')

replacements = [
    (
'''        source = inspect.getsource(TrainingService.train_v9)\n        assert 'local_structure_train_loader' in source\n        assert 'if local_structure_phase' in source\n        assert 'else parametric_train_loader' in source\n        assert 'batch_size=config.batch_size' in source\n''',
'''        source = inspect.getsource(TrainingService.train_v9)\n        assert 'authored_config.synthetic_geometry_probability = 0.0' in source\n        assert 'structural_train_dataset = PhysicalTileDatasetV9(' in source\n        assert 'if production_structure_phase:' in source\n        assert 'epoch_loader = structural_train_loader' in source\n        assert 'batch_size=config.batch_size' in source\n'''
    ),
    (
'''        assert 'self.topology_feature_project.parameters()' in structure_lock\n        assert 'self.geometry_feature_project.parameters()' in structure_lock\n        assert 'head.geometry_net.parameters()' in structure_lock\n''',
'''        assert 'self.topology_feature_project.parameters()' in structure_lock\n        assert 'self.geometry_feature_project.parameters()' in structure_lock\n        assert 'self.decoder.parameters()' in structure_lock\n        assert 'self.spline_graph.geometry_head.parameters()' in structure_lock\n'''
    ),
    (
'''        source = inspect.getsource(TrainingService.train_v9)\n        assert 'local_structure_train_dataset = ParametricPrimitiveTrainingDataset(' in source\n        assert 'batch_size=config.batch_size' in source\n        proof = source.split('if phase == "sdf-proof":', 1)[1]\n        loader_block = proof.split('elif phase == "seam-proof":', 1)[0]\n        assert 'local_structure_train_loader' in loader_block\n        assert 'train_loader if local_structure_phase' not in loader_block\n        assert '(int(config.tiles_per_epoch) + PRIMITIVE_COUNT - 1)' in source\n''',
'''        source = inspect.getsource(TrainingService.train_v9)\n        assert 'authored_config.synthetic_geometry_probability = 0.0' in source\n        assert 'structural_train_dataset = PhysicalTileDatasetV9(' in source\n        assert 'manifest, authored_config, "train", config.tiles_per_epoch' in source\n        assert 'structural_train_loader = self._build_loader(' in source\n        assert 'batch_size=config.batch_size' in source\n        assert 'if production_structure_phase:' in source\n        assert 'epoch_loader = structural_train_loader' in source\n        assert 'synthetic ladder remains validation-only' in source\n'''
    ),
    (
'''        source = inspect.getsource(TrainingService.train_v9)\n        budget = source.split('local_structure_train_tiles =', 1)[1].split(\n            'local_structure_train_dataset =', 1\n        )[0]\n        assert 'parametric_primitive_train_tiles_per_epoch' not in budget\n        assert 'PRIMITIVE_COUNT' in budget\n        loader = source.split('local_structure_train_loader = self._build_loader(', 1)[1].split(\n            'validation_loader =', 1\n        )[0]\n        assert 'batch_size=config.batch_size' in loader\n''',
'''        source = inspect.getsource(TrainingService.train_v9)\n        authored = source.split('structural_train_dataset = PhysicalTileDatasetV9(', 1)[1].split(\n            'downstream_train_dataset =', 1\n        )[0]\n        assert 'manifest, authored_config, "train", config.tiles_per_epoch' in authored\n        assert 'parametric_primitive_train_tiles_per_epoch' not in authored\n        assert 'structural_train_loader = self._build_loader(' in authored\n        assert 'batch_size=config.batch_size' in authored\n        assert 'synthetic_validation_loader = self._build_loader(' in source\n'''
    ),
    (
'''    assert 'spline = self.spline_graph(' in structure\n    assert 'self.decoder.query(' not in structure\n    assert '"spline_graph": spline["graph"]' in structure\n    assert 'self.production_structure.spline_graph.query(graph, query_grid)' in query\n''',
'''    assert 'proposal_graph = self.spline_graph.build_graph(' in structure\n    assert 'refined_graph = self.geometry_refiner(' in structure\n    assert 'self.spline_graph.query(refined_graph, query_grid)' in structure\n    assert 'self.decoder.query(' not in structure\n    assert '"spline_graph": refined_graph' in structure\n    assert 'self.production_structure.spline_graph.query(graph, query_grid)' in query\n'''
    ),
    (
'''    source = inspect.getsource(LocalBoundaryProductionContract._local_compute_losses)\n    for name in (\n        'spline_graph_topology_control', 'spline_graph_topology_sign',\n        'spline_graph_point', 'spline_graph_tangent',\n        'spline_graph_span_smoothness', 'spline_graph_span_tangent',\n        'spline_graph_span_separation', 'spline_graph_sdf',\n        'spline_graph_gradient', 'spline_graph_eikonal',\n        'spline_graph_curvature', 'spline_metric_offset',\n        'spline_metric_eikonal_near',\n    ):\n        assert f'losses["{name}"]' in source\n''',
'''    source = inspect.getsource(LocalBoundaryProductionContract._local_compute_losses)\n    for name in (\n        'spline_graph_topology_control', 'spline_graph_topology_sign',\n        'spline_graph_point', 'spline_graph_point_regret', 'spline_graph_tangent',\n        'sdf_improvement_regret', 'geometry_regret', 'boundary_pixel_regret',\n    ):\n        assert f'losses["{name}"]' in source\n    assert '(losses["spline_graph_point"] + losses["spline_graph_point_regret"])' in source\n    assert 'baseline_relative_supervision' in source\n'''
    ),
]

for old, new in replacements:
    if old not in text:
        raise SystemExit('V12 pipeline-contract anchor not found:\n' + old[:180])
    text = text.replace(old, new, 1)

path.write_text(text, encoding='utf-8')
