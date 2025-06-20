from pathlib import Path
import numpy as np


class LocalizationScneario:
    def __init__(self, npz_path: Path):
        data = np.load(npz_path, allow_pickle=True)
        self.X_true: np.ndarray = data["X_true"]
        self.full_D: np.ndarray = data["full_D"]
        self.D: np.ndarray = data["D"]
        self.B: np.ndarray = data["B"]
        self.RSS: np.ndarray = data["RSS"]
        self.num_anchors = int(data["num_anchors"])

    def get_graph(self, hop: int = 1) -> np.ndarray:
        A = self.B.copy()
        result = D.copy()
        for _ in range(hop - 1):
            A = (A @ self.B) > 0
            result = result | A
        return result.astype(int)


def generate_test(scneario_name: str):
    from dataset.generate_dataset import generate_scenario
    out_dir = Path(f"./dataset/scenarios/{scneario_name}")
    config_path = Path(f"./dataset/scenarios/{scneario_name}" + ".json")
    generate_scenario(config_path, out_dir)


if __name__ == "__main__":
    generate_test("test")
    loc = LocalizationScneario(Path("./dataset/scenarios/test/test_seed42.npz"))
    print(loc.get_graph(2))
    print(loc.X_true)