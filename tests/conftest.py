import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest

# Ensure tests import the in-repo package, not an unrelated site-packages install.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hypergraphx import Hypergraph

DATA_DIR = Path(__file__).resolve().parent.parent / "test_data"
ALL_SMALL_NUMERICAL_DATASETS = (
    "small_hypergraph1",
    "small_hypergraph2",
    "small_hypergraph3",
    "with_isolated_nodes",
)
ALL_SMALL_DATASETS = ALL_SMALL_NUMERICAL_DATASETS + ("with_literal_nodes",)
ALL_DATASETS = ALL_SMALL_DATASETS + ("justice",)


def _read_hye_list(hye_file: Path) -> List[Tuple[int]]:
    hye_list = []
    with open(hye_file, "r") as file:
        for hye in file.readlines():
            hye = hye.strip("\n").split(" ")
            hye = map(int, hye)
            hye = tuple(sorted(hye))
            hye_list.append(hye)

    counts = Counter(hye_list)
    if all(count == 1 for count in counts.values()):
        weights = None
    else:
        hye_list = list(counts.keys())
        weights = np.fromiter(counts.values(), dtype=int)

    return hye_list, weights


@pytest.fixture(scope="package")
def synthetic_small_numerical_datasets() -> Dict[str, Hypergraph]:
    datasets = dict()
    for dataset in ALL_SMALL_NUMERICAL_DATASETS:
        hye_file = DATA_DIR / dataset / "hyperedges.txt"
        hye_list, weights = _read_hye_list(hye_file)
        weighted = weights is not None
        datasets[dataset] = Hypergraph(
            edge_list=hye_list, weighted=weighted, weights=weights
        )

    return datasets


@pytest.fixture(scope="package")
def synthetic_literal_dataset() -> Dict[str, Hypergraph]:
    dataset_name = "with_literal_nodes"
    hye_file = DATA_DIR / dataset_name / "hyperedges.txt"
    with open(hye_file, "r") as file:
        hye_list = list(map(lambda hye: hye.strip("\n").split(" "), file.readlines()))
    return {dataset_name: Hypergraph(hye_list)}


@pytest.fixture(scope="package")
def justice_dataset() -> Dict[str, Hypergraph]:
    hye_list, _ = _read_hye_list(DATA_DIR / "justice_data" / "hyperedges.txt")

    with open(DATA_DIR / "justice_data" / "weights.txt") as file:
        weight_list = list(map(int, file.readlines()))

    return {"justice": Hypergraph(hye_list, weighted=True, weights=weight_list)}


@pytest.fixture(scope="package", params=ALL_DATASETS)
def loaded_hypergraph(
    synthetic_small_numerical_datasets,
    synthetic_literal_dataset,
    justice_dataset,
    request,
) -> Hypergraph:
    all_data_dict = {
        **synthetic_small_numerical_datasets,
        **synthetic_literal_dataset,
        **justice_dataset,
    }
    return all_data_dict[request.param]
