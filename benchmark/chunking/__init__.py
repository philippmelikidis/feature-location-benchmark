"""Chunking strategies for code splitting."""
from .base import BaseChunker, Chunk
from .function_level import FunctionLevelChunker
from .fixed_size import FixedSizeChunker
from .class_file_level import ClassFileLevelChunker
from .ast_chunker import ASTChunker
from .heuristic_code_aware import HeuristicCodeAwareChunker
from .virtual_document import VirtualDocumentChunker

CHUNKER_REGISTRY = {
    "function_level": FunctionLevelChunker,
    "fixed_size": FixedSizeChunker,
    "class_file_level": ClassFileLevelChunker,
    "ast_based": ASTChunker,
    "heuristic_code_aware": HeuristicCodeAwareChunker,
    "virtual_document": VirtualDocumentChunker,
}
