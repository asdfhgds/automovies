"""Storage abstraction for configurable artifact roots."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional


class StorageRoot:
    """Abstraction for project artifact storage root."""
    
    def __init__(self, root: Optional[Path] = None):
        self._root = self._resolve_root(root)
    
    def _resolve_root(self, root: Optional[Path]) -> Path:
        """Resolve storage root from various sources."""
        if root:
            return Path(root).resolve()
        
        # Check environment variable
        env_root = os.getenv("AUTOMOVIES_PROJECT_ROOT")
        if env_root:
            return Path(env_root).resolve()
        
        # Check for Google Drive mount (Colab)
        gdrive_paths = [
            Path("/content/drive/MyDrive/AutoMovies"),
            Path("/content/drive/My Drive/AutoMovies"),
            Path("/mnt/drive/MyDrive/AutoMovies"),
        ]
        for p in gdrive_paths:
            if p.exists():
                return p.resolve()
        
        # Default to local data directory
        return Path.cwd() / "data"
    
    @property
    def root(self) -> Path:
        return self._root
    
    def project_path(self, project_id: str) -> Path:
        return self._root / project_id
    
    def artifact_path(self, project_id: str, relative_path: str) -> Path:
        return self._root / project_id / relative_path
    
    def ensure_project_dir(self, project_id: str) -> Path:
        project_dir = self.project_path(project_id)
        project_dir.mkdir(parents=True, exist_ok=True)
        return project_dir
    
    def __repr__(self) -> str:
        return f"StorageRoot({self._root})"


# Global storage root instance
_storage_root: Optional[StorageRoot] = None


def get_storage_root(root: Optional[Path] = None) -> StorageRoot:
    """Get global storage root instance."""
    global _storage_root
    if _storage_root is None or root is not None:
        _storage_root = StorageRoot(root)
    return _storage_root


def set_storage_root(root: Path) -> StorageRoot:
    """Set global storage root."""
    global _storage_root
    _storage_root = StorageRoot(root)
    return _storage_root