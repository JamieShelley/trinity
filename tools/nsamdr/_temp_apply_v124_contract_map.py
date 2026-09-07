from pathlib import Path

path = Path('tools/nsamdr/neural/v9/local_boundary_production_contract.py')
text = path.read_text(encoding='utf-8')
old = '''        production = dict(contract.get("productionComponents") or {})
        production["structural representation"] = "geometry_net.production_structure"
        contract["productionComponents"] = production
'''
new = '''        production = dict(contract.get("productionComponents") or {})
        production["structural representation"] = "geometry_net.production_structure"
        production["explicit geometry refiner"] = (
            "geometry_net.production_structure.geometry_refiner"
        )
        contract["productionComponents"] = production
'''
if old not in text:
    raise SystemExit('V12.4 architecture-contract anchor not found')
text = text.replace(old, new, 1)
path.write_text(text, encoding='utf-8')


test = Path('tools/nsamdr/tests/test_v124_trainer_architecture_contract.py')
test.write_text('''from __future__ import annotations\n\nfrom pathlib import Path\nimport sys\n\nROOT = Path(__file__).resolve().parents[3]\nNEURAL = ROOT / "tools/nsamdr/neural"\nif str(NEURAL) not in sys.path:\n    sys.path.insert(0, str(NEURAL))\n\n\ndef test_v124_trainer_contract_declares_explicit_refiner():\n    from v9 import FidelityResidualNetV9, V9Config\n    from v9 import training as training_module\n    from v9.local_boundary_production_contract import (\n        install_local_boundary_training_contract,\n    )\n\n    # Reproduce the canonical trainer entrypoint ordering.  This redirects the\n    # legacy V10.7.9 validator/component map to the installed V12 local contract.\n    install_local_boundary_training_contract(training_module)\n\n    config = V9Config()\n    config.training_activation_checkpointing = False\n    model = FidelityResidualNetV9(config)\n    contract = model.architecture_contract()\n\n    production = contract["productionComponents"]\n    assert production["structural representation"] == "geometry_net.production_structure"\n    assert production["explicit geometry refiner"] == (\n        "geometry_net.production_structure.geometry_refiner"\n    )\n\n    # This is the exact trainer validator that aborted the real Raven run.\n    training_module._validate_v992_architecture_contract(contract)\n\n    modules = training_module._production_component_modules(model)\n    path, module = modules["explicit geometry refiner"]\n    assert path == "geometry_net.production_structure.geometry_refiner"\n    assert module is model.geometry_net.production_structure.geometry_refiner\n''', encoding='utf-8')
