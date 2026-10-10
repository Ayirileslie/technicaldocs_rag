"""Weaviate Cloud wrapper. The rest of the app only sees this small interface."""
import logging

import weaviate
from weaviate.classes.config import Configure, DataType, Property, Tokenization
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.util import generate_uuid5

from .config import Settings
from .models import Chunk, Hit

log = logging.getLogger(__name__)


class WeaviateStore:
    def __init__(self, settings: Settings):
        settings.require("weaviate_url", "weaviate_api_key")
        url = settings.weaviate_url if settings.weaviate_url.startswith("http") else f"https://{settings.weaviate_url}"
        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=url, auth_credentials=Auth.api_key(settings.weaviate_api_key)
        )
        self.name = settings.collection
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self.client.collections.exists(self.name):
            return
        exact = Tokenization.FIELD  # exact-match fields: no word tokenization for filters
        self.client.collections.create(
            name=self.name,
            vectorizer_config=Configure.Vectorizer.none(),  # we supply our own vectors
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="section", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT, tokenization=exact),
                Property(name="file_hash", data_type=DataType.TEXT, tokenization=exact),
                Property(name="chunk_index", data_type=DataType.INT),
            ],
        )
        log.info("created collection %s", self.name)

    @property
    def _col(self):
        return self.client.collections.get(self.name)

    def get_file_hash(self, source: str) -> str | None:
        res = self._col.query.fetch_objects(
            filters=Filter.by_property("source").equal(source), limit=1, return_properties=["file_hash"]
        )
        return res.objects[0].properties["file_hash"] if res.objects else None

    def list_sources(self) -> set[str]:
        return {o.properties["source"] for o in self._col.iterator(return_properties=["source"])}

    def delete_source(self, source: str) -> None:
        self._col.data.delete_many(where=Filter.by_property("source").equal(source))

    def replace_source(self, source: str, file_hash: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        self.delete_source(source)
        if not chunks:
            return
        col = self._col
        with col.batch.dynamic() as batch:
            for i, (c, vec) in enumerate(zip(chunks, vectors)):
                batch.add_object(
                    properties={
                        "text": c.text, "section": c.section, "source": source,
                        "file_hash": file_hash, "chunk_index": i,
                    },
                    vector=vec,
                    uuid=generate_uuid5(f"{source}:{i}"),
                )
        if col.batch.failed_objects:
            raise RuntimeError(f"{len(col.batch.failed_objects)} objects failed to insert for {source}")

    def search(self, query: str, vector: list[float], limit: int, alpha: float) -> list[Hit]:
        res = self._col.query.hybrid(
            query=query,
            vector=vector,
            alpha=alpha,
            limit=limit,
            query_properties=["text", "section"],  # BM25 over content only, not file hashes
            return_properties=["text", "source", "section"],
            return_metadata=MetadataQuery(score=True),
        )
        return [
            Hit(o.properties["text"], o.properties["source"], o.properties["section"], o.metadata.score or 0.0)
            for o in res.objects
        ]

    def close(self) -> None:
        self.client.close()