#!/usr/bin/env python3
"""
RepoDownloader - Modul zum Klonen von Git-Repositories.

Verwendung:
    from repo_downloader import RepoDownloader
    
    downloader = RepoDownloader()
    repo_path = downloader.clone("https://github.com/pallets/flask")
    python_files = downloader.get_python_files(repo_path)
"""

import subprocess
import shutil
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
import tempfile


@dataclass
class RepoInfo:
    """Informationen über ein geklontes Repository."""
    path: Path
    name: str
    url: str
    branch: str
    python_files: List[Path]
    total_files: int


class RepoDownloader:
    """Klont Git-Repositories und findet Python-Dateien."""
    
    def __init__(self, target_dir: str = None):
        """
        Initialisiert den Downloader.
        
        Args:
            target_dir: Zielverzeichnis für Repositories. 
                       Falls None, wird ein temporäres Verzeichnis verwendet.
        """
        if target_dir:
            self.target_dir = Path(target_dir)
            self.target_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.target_dir = Path(tempfile.mkdtemp(prefix="repo_"))
        
        self._check_git()
    
    def _check_git(self):
        """Prüft ob Git verfügbar ist."""
        try:
            subprocess.run(["git", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("Git ist nicht installiert oder nicht im PATH")
    
    def _extract_repo_name(self, url: str) -> str:
        """Extrahiert den Repository-Namen aus der URL."""
        # Handle verschiedene URL-Formate
        url = url.rstrip("/")
        if url.endswith(".git"):
            url = url[:-4]
        return url.split("/")[-1]
    
    def clone(self, repo_url: str, branch: str = None, depth: int = 1) -> RepoInfo:
        """
        Klont ein Git-Repository.
        
        Args:
            repo_url: URL des Repositories
            branch: Branch zum Klonen (default: main/master)
            depth: Tiefe des Klons (1 = shallow clone)
        
        Returns:
            RepoInfo mit Pfad und Dateiliste
        """
        repo_name = self._extract_repo_name(repo_url)
        repo_path = self.target_dir / repo_name
        
        # Falls bereits existiert, löschen
        if repo_path.exists():
            print(f" Repository {repo_name} existiert bereits, wird überschrieben...")
            shutil.rmtree(repo_path)
        
        # Git clone command
        cmd = ["git", "clone"]
        if depth:
            cmd.extend(["--depth", str(depth)])
        if branch:
            cmd.extend(["--branch", branch])
        cmd.extend([repo_url, str(repo_path)])
        
        print(f"Klone {repo_url}...")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"Repository geklont nach {repo_path}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Git clone fehlgeschlagen: {e.stderr}")
        
        # Branch ermitteln falls nicht angegeben
        if not branch:
            try:
                result = subprocess.run(
                    ["git", "-C", str(repo_path), "rev-parse", "--abbrev-ref", "HEAD"],
                    capture_output=True, text=True, check=True
                )
                branch = result.stdout.strip()
            except:
                branch = "unknown"
        
        # Python-Dateien finden
        python_files = self.get_python_files(repo_path)
        total_files = len(list(repo_path.rglob("*")))
        
        return RepoInfo(
            path=repo_path,
            name=repo_name,
            url=repo_url,
            branch=branch,
            python_files=python_files,
            total_files=total_files
        )
    
    def get_python_files(self, repo_path: Path, exclude_tests: bool = False,
                         exclude_patterns: List[str] = None) -> List[Path]:
        """
        Findet alle Python-Dateien in einem Repository.
        
        Args:
            repo_path: Pfad zum Repository
            exclude_tests: Tests ausschließen
            exclude_patterns: Zusätzliche Ausschlussmuster
        
        Returns:
            Liste von Python-Dateipfaden
        """
        exclude_patterns = exclude_patterns or []
        
        # Standard-Ausschlüsse
        default_excludes = [
            ".git", "__pycache__", ".tox", ".eggs", "*.egg-info",
            "build", "dist", ".venv", "venv", "env", ".env",
            "node_modules", ".pytest_cache", ".mypy_cache"
        ]
        
        if exclude_tests:
            default_excludes.extend(["test", "tests", "test_*.py", "*_test.py"])
        
        all_excludes = set(default_excludes + exclude_patterns)
        
        python_files = []
        for py_file in repo_path.rglob("*.py"):
            # Prüfen ob Datei in ausgeschlossenem Pfad liegt
            path_parts = py_file.relative_to(repo_path).parts
            
            skip = False
            for part in path_parts:
                for exclude in all_excludes:
                    if exclude.startswith("*"):
                        if part.endswith(exclude[1:]):
                            skip = True
                            break
                    elif part == exclude or part.startswith(exclude):
                        skip = True
                        break
                if skip:
                    break
            
            if not skip:
                python_files.append(py_file)
        
        # Sortieren für konsistente Reihenfolge
        python_files.sort()
        
        return python_files
    
    def cleanup(self, repo_path: Path = None):
        """
        Löscht ein geklontes Repository.
        
        Args:
            repo_path: Pfad zum Repository. Falls None, wird das gesamte Zielverzeichnis gelöscht.
        """
        if repo_path:
            if repo_path.exists():
                shutil.rmtree(repo_path)
                print(f" Repository gelöscht: {repo_path}")
        else:
            if self.target_dir.exists():
                shutil.rmtree(self.target_dir)
                print(f" Zielverzeichnis gelöscht: {self.target_dir}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Optional: Cleanup bei Context-Manager-Exit
        pass


# Test
if __name__ == "__main__":
    import sys
    
    # Test mit kleinem Repo
    test_url = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/pallets/click"
    
    with RepoDownloader() as downloader:
        print(f"\n=== Test: {test_url} ===\n")
        
        try:
            info = downloader.clone(test_url, depth=1)
            
            print(f"\nRepository Info:")
            print(f"   Name: {info.name}")
            print(f"   Branch: {info.branch}")
            print(f"   Pfad: {info.path}")
            print(f"   Gesamt Dateien: {info.total_files}")
            print(f"   Python Dateien: {len(info.python_files)}")
            
            if info.python_files:
                print(f"\nErste 10 Python-Dateien:")
                for f in info.python_files[:10]:
                    rel_path = f.relative_to(info.path)
                    print(f"   - {rel_path}")
                if len(info.python_files) > 10:
                    print(f"   ... und {len(info.python_files) - 10} weitere")
            
            # Cleanup
            # downloader.cleanup(info.path)
            print(f"\nRepository bleibt unter: {info.path}")
            
        except Exception as e:
            print(f"Fehler: {e}")
            sys.exit(1)
