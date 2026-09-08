import unittest

from lab_bench.core import HostInfo, ModelInfo
from lab_bench.planner import Candidate, plan_models


def candidate(name, weight, roles, stage=1):
    return Candidate(
        name=name,
        display_name=name,
        estimated_weight_gb=weight,
        min_context=65536,
        german=True,
        tools=True,
        roles=roles,
        cpu_stage=stage,
        gpu_stage=stage,
        cpu_priority=10,
        gpu_priority=10,
        source="test",
    )


class PlannerTests(unittest.TestCase):
    def test_cpu_profile_selects_small_coverage_set(self):
        host = HostInfo("cpu", "Linux", "test", "CPU", 8, 32768, [], "http://localhost")
        candidates = [
            candidate("small", 3, ["baseline", "german"]),
            candidate("medium", 8, ["scaling"]),
            candidate("too-large", 30, ["quality-ceiling"]),
        ]
        planned = plan_models(host, candidates, [])
        names = {item.candidate.name for item in planned}
        self.assertIn("small", names)
        self.assertNotIn("too-large", names)

    def test_gpu_profile_allows_selected_offload(self):
        host = HostInfo(
            "gpu",
            "Linux",
            "test",
            "CPU",
            8,
            32768,
            [{"memory_total_mib": 12288}],
            "http://localhost",
        )
        candidates = [
            candidate("native", 7, ["baseline"]),
            candidate("offload", 18, ["coding"]),
            candidate("stage-two", 18, ["quality-ceiling"], stage=2),
        ]
        planned = plan_models(host, candidates, [], stage=1)
        names = {item.candidate.name for item in planned}
        self.assertIn("native", names)
        self.assertIn("offload", names)
        self.assertNotIn("stage-two", names)

    def test_installed_models_are_marked(self):
        host = HostInfo("cpu", "Linux", "test", "CPU", 8, 32768, [], "http://localhost")
        result = plan_models(host, [candidate("small", 3, ["baseline"])], [ModelInfo(name="small", available=True)])
        self.assertTrue(result[0].installed)


if __name__ == "__main__":
    unittest.main()