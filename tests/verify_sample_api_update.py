from syntho_hive.interface.synthesizer import Synthesizer
from syntho_hive.interface.config import Metadata, PrivacyConfig
from unittest.mock import MagicMock
import pandas as pd
import sys

try:
    meta = MagicMock(spec=Metadata)
    t_config = MagicMock()
    t_config.fk = {}
    t_config.pk = "id"
    t_config.parent_context_cols = []
    t_config.has_dependencies = False
    
    # Metadata.tables is a dict of TableConfigs
    meta.tables = {"users": t_config, "orders": t_config}
    # Metadata.get_table returns the config
    meta.get_table.return_value = t_config

    privacy = MagicMock(spec=PrivacyConfig)
    spark = MagicMock()

    # We need to patch SchemaGraph so it doesn't fail on our mock metadata if it does deep inspection
    # But StagedOrchestrator init creates SchemaGraph(metadata).
    # Let's just mock StagedOrchestrator class entirely so Synthesizer doesn't instantiate the real one.
    
    # Actually, easier to just instantiate Synthesizer with spark=None to avoid Orchestrator init?
    # No, Synthesizer checks "if self.spark: self.orchestrator = ... else None"
    # But then sample() checks "if not self.orchestrator: raise ValueError"
    
    # So we MUST provide spark, which triggers Orchestrator init.
    # To avoid Orchestrator init failing, we should patch `syntho_hive.interface.synthesizer.StagedOrchestrator`
    
    from unittest.mock import patch
    
    with patch("syntho_hive.interface.synthesizer.StagedOrchestrator") as MockOrchestrator:
        synth = Synthesizer(meta, privacy, spark_session=spark)
        
        # Synthesizer instance has synth.orchestrator as the instance of MockOrchestrator
        # We need to set return value of generate on that instance
        
        # When output_path_base is None, return dict of dfs
        mock_dfs = {"users": pd.DataFrame({"id": [1]}), "orders": pd.DataFrame({"id": [10]})}
        synth.orchestrator.generate.return_value = mock_dfs

        # 2. Test output_path=None
        print("Running Test 1: output_path=None")
        dfs = synth.sample(num_rows={"users": 10}, output_path=None)
        
        # Check that orchestrator was called with None
        synth.orchestrator.generate.assert_called_with({"users": 10}, output_path_base=None)
        
        assert isinstance(dfs, dict), f"Expected dict, got {type(dfs)}"
        assert "users" in dfs
        assert isinstance(dfs["users"], pd.DataFrame), "Expected DataFrame value"
        print("Test 1 Passed: Output Path None returns DF Dict")

        # 3. Test output_path="some/path"
        print("Running Test 2: output_path set")
        synth.orchestrator.generate.return_value = {"users": pd.DataFrame()} # It still returns dict now
        paths = synth.sample(num_rows={"users": 10}, output_path="/tmp/test")
        
        # Check orchestrator called with path
        synth.orchestrator.generate.assert_called_with({"users": 10}, "/tmp/test")
        
        assert isinstance(paths, dict)
        assert paths["users"] == "/tmp/test/users"
        print("Test 2 Passed: Output Path set returns Path Dict")

    # If we get here, pass.
    sys.exit(0)


except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)
