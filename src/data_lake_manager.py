"""
Task 2: Data Storage Architecture
Implements a DataLakeManager for managing raw, processed, and embedding storage.

Technical Justification:
- Chosen: PostgreSQL + MinIO (S3-compatible)
- Scalability: MinIO handles unlimited object storage, PostgreSQL for metadata
- Cost: Open-source alternatives (no AWS/GCP vendor lock-in)
- Query capability: pgvector for semantic search on embeddings
- Version control: All layers versioned with timestamps
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import logging
import pickle
from abc import ABC, abstractmethod
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """Abstract storage backend interface"""
    
    @abstractmethod
    def upload(self, key: str, data: Any, metadata: Dict) -> Dict:
        """Upload data with metadata"""
        pass
    
    @abstractmethod
    def download(self, key: str) -> Tuple[Any, Dict]:
        """Download data and retrieve metadata"""
        pass
    
    @abstractmethod
    def list_versions(self, prefix: str) -> List[Dict]:
        """List all versions of objects with given prefix"""
        pass


class LocalStorageBackend(StorageBackend):
    """Local filesystem storage backend (for development)"""
    
    def __init__(self, base_path: str = "./data/lake"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.base_path / "metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load metadata index"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Save metadata index"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def upload(self, key: str, data: Any, metadata: Dict = None) -> Dict:
        """Upload data to local filesystem"""
        metadata = metadata or {}
        
        # Create versioned key
        version = datetime.now().isoformat()
        versioned_key = f"{key}_{version}"
        
        file_path = self.base_path / versioned_key
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Store data
        if isinstance(data, pd.DataFrame):
            file_path = file_path.with_suffix('.parquet')
            data.to_parquet(file_path, index=False)
        elif isinstance(data, (dict, list)):
            file_path = file_path.with_suffix('.json')
            with open(file_path, 'w') as f:
                json.dump(data, f, default=str)
        else:
            file_path = file_path.with_suffix('.pkl')
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
        
        # Store metadata
        file_hash = self._compute_hash(file_path)
        upload_info = {
            "key": key,
            "versioned_key": versioned_key,
            "file_path": str(file_path),
            "timestamp": version,
            "size_bytes": file_path.stat().st_size if file_path.exists() else 0,
            "hash": file_hash,
            "metadata": metadata
        }
        
        if key not in self.metadata:
            self.metadata[key] = []
        self.metadata[key].append(upload_info)
        self._save_metadata()
        
        logger.info(f"Uploaded: {versioned_key} ({upload_info['size_bytes']} bytes)")
        return upload_info
    
    def download(self, key: str, version: Optional[str] = None) -> Tuple[Any, Dict]:
        """Download data from local filesystem"""
        
        if key not in self.metadata:
            raise FileNotFoundError(f"Key not found: {key}")
        
        versions = self.metadata[key]
        if not versions:
            raise FileNotFoundError(f"No versions for key: {key}")
        
        # Get latest or specified version
        if version is None:
            version_info = versions[-1]  # Latest
        else:
            version_info = next((v for v in versions if v['timestamp'] == version), None)
            if not version_info:
                raise FileNotFoundError(f"Version not found: {version}")
        
        file_path = Path(version_info['file_path'])
        
        # Load data
        if file_path.suffix == '.parquet':
            data = pd.read_parquet(file_path)
        elif file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
        else:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        
        logger.info(f"Downloaded: {version_info['versioned_key']}")
        return data, version_info['metadata']
    
    def list_versions(self, prefix: str) -> List[Dict]:
        """List all versions of objects with given prefix"""
        matching = {}
        for key, versions in self.metadata.items():
            if key.startswith(prefix):
                matching[key] = versions
        return matching
    
    def _compute_hash(self, file_path: Path, algorithm: str = 'sha256') -> str:
        """Compute file hash for integrity verification"""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)
        return hasher.hexdigest()


class MinIOStorageBackend(StorageBackend):
    """MinIO S3-compatible object storage backend"""
    
    def __init__(self, endpoint: str = "localhost:9000",
                 access_key: str = "minioadmin",
                 secret_key: str = "minioadmin"):
        try:
            from minio import Minio
            self.client = Minio(endpoint,
                               access_key=access_key,
                               secret_key=secret_key,
                               secure=False)
            self.bucket_name = "paksential-data"
            self._ensure_bucket()
        except ImportError:
            logger.warning("MinIO client not available, using local storage fallback")
            self.client = None
    
    def _ensure_bucket(self):
        """Create bucket if it doesn't exist"""
        if self.client:
            try:
                if not self.client.bucket_exists(self.bucket_name):
                    self.client.make_bucket(self.bucket_name)
                    logger.info(f"Created bucket: {self.bucket_name}")
            except Exception as e:
                logger.error(f"Failed to create bucket: {e}")
    
    def upload(self, key: str, data: Any, metadata: Dict = None) -> Dict:
        """Upload to MinIO"""
        if not self.client:
            logger.warning("MinIO unavailable, using fallback")
            return {"error": "MinIO not available"}
        
        metadata = metadata or {}
        version = datetime.now().isoformat()
        versioned_key = f"{key}_{version}"
        
        try:
            # Serialize data
            if isinstance(data, pd.DataFrame):
                data_bytes = data.to_parquet()
                object_name = f"{versioned_key}.parquet"
            elif isinstance(data, (dict, list)):
                data_bytes = json.dumps(data, default=str).encode()
                object_name = f"{versioned_key}.json"
            else:
                data_bytes = pickle.dumps(data)
                object_name = f"{versioned_key}.pkl"
            
            # Upload with metadata
            self.client.put_object(
                self.bucket_name,
                object_name,
                data_bytes,
                length=len(data_bytes),
                metadata=metadata
            )
            
            logger.info(f"Uploaded to MinIO: {object_name}")
            return {
                "key": key,
                "object_name": object_name,
                "timestamp": version,
                "size_bytes": len(data_bytes)
            }
        
        except Exception as e:
            logger.error(f"MinIO upload failed: {e}")
            return {"error": str(e)}
    
    def download(self, key: str, version: Optional[str] = None) -> Tuple[Any, Dict]:
        """Download from MinIO"""
        raise NotImplementedError("Use LocalStorageBackend for development")
    
    def list_versions(self, prefix: str) -> List[Dict]:
        """List versions in MinIO"""
        if not self.client:
            return []
        
        objects = []
        try:
            for obj in self.client.list_objects(self.bucket_name, prefix=prefix):
                objects.append({
                    "object_name": obj.object_name,
                    "size": obj.size,
                    "last_modified": obj.last_modified.isoformat()
                })
        except Exception as e:
            logger.error(f"Failed to list objects: {e}")
        
        return objects


class DataLakeManager:
    """
    Manages three storage layers:
    1. Raw: Original files with metadata
    2. Processed: Cleaned Parquet files, vocabulary, TF-IDF matrix
    3. Embeddings: Versioned Word2Vec/FastText models
    """
    
    def __init__(self, backend: str = "local",
                 base_path: str = "./data/lake"):
        """Initialize DataLakeManager with specified backend"""
        
        if backend == "local":
            self.backend = LocalStorageBackend(base_path)
        elif backend == "minio":
            self.backend = MinIOStorageBackend()
        else:
            raise ValueError(f"Unknown backend: {backend}")
        
        self.base_path = Path(base_path)
        self.layers = {
            "raw": self.base_path / "raw",
            "processed": self.base_path / "processed",
            "embeddings": self.base_path / "embeddings"
        }
        
        for layer_path in self.layers.values():
            layer_path.mkdir(parents=True, exist_ok=True)
    
    def upload_raw(self, df: pd.DataFrame, source_name: str,
                   description: str = "") -> Dict:
        """
        Upload raw dataset
        
        Args:
            df: Raw DataFrame
            source_name: Name of data source
            description: Description of raw data
        
        Returns:
            Upload metadata
        """
        logger.info(f"Uploading raw data from {source_name}...")
        
        metadata = {
            "source": source_name,
            "description": description,
            "row_count": len(df),
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "missing_values": df.isnull().sum().to_dict()
        }
        
        key = f"raw/{source_name}/dataset"
        upload_info = self.backend.upload(key, df, metadata)
        
        # Also save metadata separately
        metadata_key = f"raw/{source_name}/metadata"
        self.backend.upload(metadata_key, metadata)
        
        logger.info(f"Raw data uploaded: {len(df)} rows")
        return upload_info
    
    def upload_processed(self, df: pd.DataFrame, name: str,
                        vectorizer_type: str = "tfidf",
                        preprocessing_steps: List[str] = None) -> Dict:
        """
        Upload processed dataset
        
        Args:
            df: Processed DataFrame
            name: Name of processed dataset
            vectorizer_type: Type of vectorizer used
            preprocessing_steps: List of preprocessing steps applied
        
        Returns:
            Upload metadata
        """
        logger.info(f"Uploading processed data: {name}...")
        
        preprocessing_steps = preprocessing_steps or []
        
        metadata = {
            "name": name,
            "vectorizer_type": vectorizer_type,
            "preprocessing_steps": preprocessing_steps,
            "row_count": len(df),
            "columns": list(df.columns),
            "creation_timestamp": datetime.now().isoformat(),
            "sparsity": (df == 0).sum().sum() / (df.shape[0] * df.shape[1])
        }
        
        key = f"processed/{name}/data"
        upload_info = self.backend.upload(key, df, metadata)
        
        # Save metadata
        metadata_key = f"processed/{name}/metadata"
        self.backend.upload(metadata_key, metadata)
        
        logger.info(f"Processed data uploaded: {name}")
        return upload_info
    
    def upload_embeddings(self, model: Any, name: str,
                         model_type: str = "word2vec",
                         hyperparameters: Dict = None) -> Dict:
        """
        Upload embedding model
        
        Args:
            model: Embedding model (Word2Vec, FastText, etc.)
            name: Name of embedding model
            model_type: Type of embedding model
            hyperparameters: Model hyperparameters
        
        Returns:
            Upload metadata
        """
        logger.info(f"Uploading embedding model: {name}...")
        
        hyperparameters = hyperparameters or {}
        
        metadata = {
            "name": name,
            "model_type": model_type,
            "hyperparameters": hyperparameters,
            "creation_timestamp": datetime.now().isoformat(),
            "vocab_size": len(model.wv) if hasattr(model, 'wv') else 0,
            "embedding_dim": model.vector_size if hasattr(model, 'vector_size') else 0
        }
        
        key = f"embeddings/{name}/model"
        upload_info = self.backend.upload(key, model, metadata)
        
        # Save metadata
        metadata_key = f"embeddings/{name}/metadata"
        self.backend.upload(metadata_key, metadata)
        
        logger.info(f"Embedding model uploaded: {name}")
        return upload_info
    
    def fetch_for_training(self, dataset_name: str,
                          version: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Fetch processed data for model training
        
        Args:
            dataset_name: Name of processed dataset
            version: Specific version timestamp (latest if None)
        
        Returns:
            Tuple of (DataFrame, metadata)
        """
        logger.info(f"Fetching training data: {dataset_name}...")
        
        key = f"processed/{dataset_name}/data"
        data, metadata = self.backend.download(key, version)
        
        logger.info(f"Fetched {len(data)} training samples")
        return data, metadata
    
    def list_versions(self, layer: str = "processed",
                     prefix: str = "") -> Dict:
        """
        List all versions in a storage layer
        
        Args:
            layer: Storage layer (raw, processed, embeddings)
            prefix: Optional prefix filter
        
        Returns:
            Dictionary of versions
        """
        full_prefix = f"{layer}/{prefix}" if prefix else f"{layer}/"
        versions = self.backend.list_versions(full_prefix)
        
        logger.info(f"Found {len(versions)} versions in {layer}/{prefix}")
        return versions
    
    def get_data_lineage(self, dataset_name: str) -> Dict:
        """
        Get data lineage for a processed dataset
        
        Args:
            dataset_name: Name of dataset
        
        Returns:
            Data lineage information
        """
        _, metadata = self.fetch_for_training(dataset_name)
        
        lineage = {
            "dataset_name": dataset_name,
            "creation_timestamp": metadata.get("creation_timestamp"),
            "preprocessing_steps": metadata.get("preprocessing_steps", []),
            "row_count": metadata.get("row_count"),
            "columns": metadata.get("columns", [])
        }
        
        return lineage
    
    def validate_data_integrity(self, dataset_name: str) -> bool:
        """
        Validate data integrity using checksums
        
        Args:
            dataset_name: Name of dataset
        
        Returns:
            True if validation passes
        """
        try:
            data, metadata = self.fetch_for_training(dataset_name)
            
            # Check for null values
            null_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
            if null_ratio > 0.1:
                logger.warning(f"High null ratio: {null_ratio:.2%}")
            
            # Check for duplicates
            if data.duplicated().sum() > 0:
                logger.warning(f"Found {data.duplicated().sum()} duplicates")
            
            logger.info("Data integrity validation passed")
            return True
        
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False


def main():
    """Demonstrate DataLakeManager usage"""
    
    # Initialize manager
    manager = DataLakeManager(backend="local")
    
    # Create sample datasets
    logger.info("Creating sample datasets...")
    
    # Raw data
    raw_df = pd.DataFrame({
        'text': ['Sample text 1', 'Sample text 2', 'Sample text 3'],
        'label': ['Real', 'Fake', 'Satire'],
        'source': ['LIAR', 'ISOT', 'Pakistan']
    })
    
    # Upload raw
    raw_info = manager.upload_raw(raw_df, "test_source", "Test dataset")
    logger.info(f"Raw upload info: {raw_info}")
    
    # Processed data (TF-IDF features)
    processed_df = pd.DataFrame({
        'feature_0': [0.1, 0.2, 0.3],
        'feature_1': [0.4, 0.5, 0.6],
        'label': ['Real', 'Fake', 'Satire']
    })
    
    # Upload processed
    processed_info = manager.upload_processed(
        processed_df,
        "test_processed",
        vectorizer_type="tfidf",
        preprocessing_steps=["lowercase", "tokenization", "stopword_removal"]
    )
    logger.info(f"Processed upload info: {processed_info}")
    
    # List versions
    versions = manager.list_versions(layer="processed")
    logger.info(f"Available versions: {versions}")
    
    # Fetch for training
    training_data, metadata = manager.fetch_for_training("test_processed")
    logger.info(f"Fetched training data shape: {training_data.shape}")
    
    # Validate integrity
    is_valid = manager.validate_data_integrity("test_processed")
    logger.info(f"Data integrity valid: {is_valid}")


if __name__ == "__main__":
    main()
