from __future__ import annotations


class ModelVersionNotFoundError(Exception):
    # Raised when the requested model version is not present in the local registry.
    def __init__(self, requested_version: str, available_versions: list[str]):
        self.requested_version = requested_version
        self.available_versions = available_versions
        super().__init__(requested_version)
