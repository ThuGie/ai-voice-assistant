import unittest
from unittest.mock import patch, MagicMock, mock_open
import numpy as np
from src.rag import RAG

class TestRAG(unittest.TestCase):

    def setUp(self):
        self.rag_available_patcher = patch('src.rag.RAG_DEPENDENCIES_AVAILABLE', True)
        self.chromadb_patcher = patch('src.rag.chromadb')
        # Use spec=True to make the mock stricter
        self.st_patcher = patch('src.rag.SentenceTransformer', spec=True)

        self.mock_rag_available = self.rag_available_patcher.start()
        self.mock_chromadb = self.chromadb_patcher.start()
        self.mock_sentence_transformer = self.st_patcher.start()

    def tearDown(self):
        self.rag_available_patcher.stop()
        self.chromadb_patcher.stop()
        self.st_patcher.stop()

    def test_add_document_success(self):
        # Configure the mocks
        mock_st_instance = self.mock_sentence_transformer.return_value
        mock_st_instance.encode = MagicMock(return_value=np.array([[0.1, 0.2, 0.3]]))

        mock_collection = self.mock_chromadb.PersistentClient.return_value.get_or_create_collection.return_value

        # Create the RAG instance
        rag = RAG()

        # Check if initialization was successful
        self.assertTrue(rag.is_available)

        # Mock the open call
        m = mock_open(read_data="This is a test document.")
        with patch("builtins.open", m):
            result = rag.add_document("test.txt")

        # Assert the results
        self.assertTrue(result.startswith("Successfully added document"))
        mock_collection.add.assert_called_once()
        mock_st_instance.encode.assert_called_once()


class TestRAGNoDependencies(unittest.TestCase):

    def test_rag_init_no_dependencies(self):
        with patch('src.rag.RAG_DEPENDENCIES_AVAILABLE', False):
            from src.rag import RAG
            rag = RAG()
            self.assertFalse(rag.is_available)

if __name__ == '__main__':
    unittest.main()
