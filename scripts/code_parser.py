#!/usr/bin/env python3
"""
CodeParser - Tree-sitter basierter Python-Code-Parser.

Extrahiert semantische Einheiten (Module, Klassen, Funktionen) aus Python-Code.

Verwendung:
    from code_parser import CodeParser
    
    parser = CodeParser()
    units = parser.parse_file(Path("example.py"))
    for unit in units:
        print(f"{unit.type}: {unit.name}")
"""

import tree_sitter_python as ts_python
from tree_sitter import Language, Parser, Node
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class CodeUnit:
    """Repräsentiert eine semantische Code-Einheit."""
    type: str              # "module", "class", "function", "method"
    name: str              # Name der Einheit
    qualified_name: str    # Vollqualifizierter Name (z.B. "MyClass.my_method")
    docstring: str         # Docstring falls vorhanden
    signature: str         # Funktions-/Klassensignatur
    body: str              # Vollständiger Code
    start_line: int        # Startzeile (1-indexed)
    end_line: int          # Endzeile (1-indexed)
    parent: Optional[str]  # Parent-Klasse falls Methode
    file_path: str         # Ursprüngliche Datei
    imports: List[str] = field(default_factory=list)     # Imports (nur für Module)
    decorators: List[str] = field(default_factory=list)  # Decorators


class CodeParser:
    """Parser für Python-Code mittels Tree-sitter."""
    
    def __init__(self):
        """Initialisiert den Parser mit Python-Grammatik."""
        self.parser = Parser()
        self.parser.language = Language(ts_python.language())
    
    def parse_file(self, file_path: Path) -> List[CodeUnit]:
        """
        Parst eine Python-Datei und extrahiert Code-Einheiten.
        
        Args:
            file_path: Pfad zur Python-Datei
        
        Returns:
            Liste von CodeUnit-Objekten
        """
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f" Fehler beim Lesen von {file_path}: {e}")
            return []
        
        return self.parse_content(content, str(file_path))
    
    def parse_content(self, content: str, file_path: str = "<string>") -> List[CodeUnit]:
        """
        Parst Python-Code-String und extrahiert Code-Einheiten.
        
        Args:
            content: Python-Quellcode
            file_path: Dateipfad für Referenz
        
        Returns:
            Liste von CodeUnit-Objekten
        """
        tree = self.parser.parse(bytes(content, "utf-8"))
        root = tree.root_node
        
        units = []
        lines = content.split("\n")
        
        # Modul-Level Docstring und Imports extrahieren
        module_docstring = self._extract_module_docstring(root, lines)
        imports = self._extract_imports(root, lines)
        
        # Modul als erste Einheit
        module_name = Path(file_path).stem if file_path != "<string>" else "module"
        units.append(CodeUnit(
            type="module",
            name=module_name,
            qualified_name=module_name,
            docstring=module_docstring,
            signature=f"# Module: {module_name}",
            body=content,
            start_line=1,
            end_line=len(lines),
            parent=None,
            file_path=file_path,
            imports=imports,
            decorators=[]
        ))
        
        # Top-Level Klassen und Funktionen
        for child in root.children:
            if child.type == "class_definition":
                class_unit = self._parse_class(child, lines, file_path, None)
                if class_unit:
                    units.append(class_unit)
                    # Methoden der Klasse
                    units.extend(self._parse_class_methods(child, lines, file_path, class_unit.name))
            
            elif child.type == "function_definition":
                func_unit = self._parse_function(child, lines, file_path, None)
                if func_unit:
                    units.append(func_unit)
            
            elif child.type == "decorated_definition":
                # Decorated class or function
                decorators = self._extract_decorators(child, lines)
                for inner in child.children:
                    if inner.type == "class_definition":
                        class_unit = self._parse_class(inner, lines, file_path, None, decorators)
                        if class_unit:
                            units.append(class_unit)
                            units.extend(self._parse_class_methods(inner, lines, file_path, class_unit.name))
                    elif inner.type == "function_definition":
                        func_unit = self._parse_function(inner, lines, file_path, None, decorators)
                        if func_unit:
                            units.append(func_unit)
        
        return units
    
    def _parse_class(self, node: Node, lines: List[str], file_path: str, 
                     parent: Optional[str], decorators: List[str] = None) -> Optional[CodeUnit]:
        """Parst eine Klassendefinition."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return None
        
        name = self._get_node_text(name_node, lines)
        docstring = self._extract_docstring(node, lines)
        signature = self._extract_signature(node, lines)
        body = self._get_node_text(node, lines)
        
        return CodeUnit(
            type="class",
            name=name,
            qualified_name=f"{parent}.{name}" if parent else name,
            docstring=docstring,
            signature=signature,
            body=body,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            parent=parent,
            file_path=file_path,
            imports=[],
            decorators=decorators or []
        )
    
    def _parse_function(self, node: Node, lines: List[str], file_path: str,
                        parent: Optional[str], decorators: List[str] = None) -> Optional[CodeUnit]:
        """Parst eine Funktionsdefinition."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return None
        
        name = self._get_node_text(name_node, lines)
        docstring = self._extract_docstring(node, lines)
        signature = self._extract_signature(node, lines)
        body = self._get_node_text(node, lines)
        
        unit_type = "method" if parent else "function"
        
        return CodeUnit(
            type=unit_type,
            name=name,
            qualified_name=f"{parent}.{name}" if parent else name,
            docstring=docstring,
            signature=signature,
            body=body,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            parent=parent,
            file_path=file_path,
            imports=[],
            decorators=decorators or []
        )
    
    def _parse_class_methods(self, class_node: Node, lines: List[str], 
                              file_path: str, class_name: str) -> List[CodeUnit]:
        """Extrahiert alle Methoden einer Klasse."""
        methods = []
        
        body_node = class_node.child_by_field_name("body")
        if not body_node:
            return methods
        
        for child in body_node.children:
            if child.type == "function_definition":
                method = self._parse_function(child, lines, file_path, class_name)
                if method:
                    methods.append(method)
            elif child.type == "decorated_definition":
                decorators = self._extract_decorators(child, lines)
                for inner in child.children:
                    if inner.type == "function_definition":
                        method = self._parse_function(inner, lines, file_path, class_name, decorators)
                        if method:
                            methods.append(method)
        
        return methods
    
    def _extract_module_docstring(self, root: Node, lines: List[str]) -> str:
        """Extrahiert Modul-Docstring."""
        for child in root.children:
            if child.type == "expression_statement":
                for sub in child.children:
                    if sub.type == "string":
                        return self._clean_docstring(self._get_node_text(sub, lines))
            # Überspringe Kommentare und leere Zeilen
            elif child.type not in ("comment", ""):
                break
        return ""
    
    def _extract_docstring(self, node: Node, lines: List[str]) -> str:
        """Extrahiert Docstring aus Funktion/Klasse."""
        body_node = node.child_by_field_name("body")
        if body_node and body_node.children:
            first_stmt = body_node.children[0]
            if first_stmt.type == "expression_statement":
                for sub in first_stmt.children:
                    if sub.type == "string":
                        return self._clean_docstring(self._get_node_text(sub, lines))
        return ""
    
    def _extract_signature(self, node: Node, lines: List[str]) -> str:
        """Extrahiert die Signatur (erste Zeile) einer Definition."""
        start_line = node.start_point[0]
        # Finde Zeile mit ":"
        for i in range(start_line, min(start_line + 5, len(lines))):
            line = lines[i]
            if ":" in line:
                return line.strip()
        return lines[start_line].strip() if start_line < len(lines) else ""
    
    def _extract_imports(self, root: Node, lines: List[str]) -> List[str]:
        """Extrahiert alle Import-Statements."""
        imports = []
        for child in root.children:
            if child.type in ("import_statement", "import_from_statement"):
                imports.append(self._get_node_text(child, lines))
        return imports
    
    def _extract_decorators(self, decorated_node: Node, lines: List[str]) -> List[str]:
        """Extrahiert Decorators von einer decorated_definition."""
        decorators = []
        for child in decorated_node.children:
            if child.type == "decorator":
                decorators.append(self._get_node_text(child, lines))
        return decorators
    
    def _get_node_text(self, node: Node, lines: List[str]) -> str:
        """Extrahiert Text eines Nodes."""
        start_row, start_col = node.start_point
        end_row, end_col = node.end_point
        
        if start_row == end_row:
            return lines[start_row][start_col:end_col]
        
        result = [lines[start_row][start_col:]]
        for i in range(start_row + 1, end_row):
            result.append(lines[i])
        result.append(lines[end_row][:end_col])
        
        return "\n".join(result)
    
    def _clean_docstring(self, docstring: str) -> str:
        """Bereinigt einen Docstring (entfernt Quotes)."""
        ds = docstring.strip()
        for quote in ('"""', "'''", '"', "'"):
            if ds.startswith(quote) and ds.endswith(quote):
                ds = ds[len(quote):-len(quote)]
                break
        return ds.strip()
    
    def parse_directory(self, dir_path: Path, exclude_tests: bool = False) -> List[CodeUnit]:
        """
        Parst alle Python-Dateien in einem Verzeichnis.
        
        Args:
            dir_path: Pfad zum Verzeichnis
            exclude_tests: Test-Dateien ausschließen
        
        Returns:
            Liste aller CodeUnit-Objekte
        """
        all_units = []
        
        for py_file in sorted(dir_path.rglob("*.py")):
            # Ausschlüsse
            rel_parts = py_file.relative_to(dir_path).parts
            skip = False
            for part in rel_parts:
                if part in ("__pycache__", ".git", ".tox", "build", "dist", ".venv", "venv"):
                    skip = True
                    break
                if exclude_tests and part in ("test", "tests"):
                    skip = True
                    break
            
            if not skip:
                units = self.parse_file(py_file)
                all_units.extend(units)
        
        return all_units


# Test
if __name__ == "__main__":
    import sys
    
    parser = CodeParser()
    
    # Test mit einer Datei oder Inhalt
    test_code = '''
"""Module docstring."""

import os
from typing import List

@dataclass
class MyClass:
    """Class docstring."""
    
    def __init__(self, name: str):
        """Init docstring."""
        self.name = name
    
    @property
    def upper_name(self) -> str:
        """Property docstring."""
        return self.name.upper()


def my_function(x: int, y: int) -> int:
    """Function docstring."""
    return x + y
'''
    
    print("=== Parsing Test Code ===\n")
    units = parser.parse_content(test_code, "test.py")
    
    for unit in units:
        print(f"[{unit.type.upper()}] {unit.qualified_name}")
        print(f"  Lines: {unit.start_line}-{unit.end_line}")
        if unit.docstring:
            print(f"  Docstring: {unit.docstring[:50]}...")
        if unit.decorators:
            print(f"  Decorators: {unit.decorators}")
        print()
    
    print(f"\n{len(units)} Einheiten gefunden")
