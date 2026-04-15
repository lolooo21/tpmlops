from __future__ import annotations

from api.repositories.test_data_repository import TestDataRepository


class GetRandomTestSampleUseCase:
    # This use case exists only for the debug endpoint and stays outside prediction flow.
    def __init__(self, test_data_repository: TestDataRepository):
        self._test_data_repository = test_data_repository

    def execute(self) -> tuple[dict, float]:
        return self._test_data_repository.get_random_test_row()
