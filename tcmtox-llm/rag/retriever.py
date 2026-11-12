# -*- coding: utf-8 -*-
"""
检索客户端：Embedding/Rerank provider 抽象 + 向量存储 + 语义检索

设计要点（见 docs/RAG检索层设计.md）：
  - EmbeddingProvider 二选一：local（BGE-m3 本地）/ api（newapi qwen3-embedding）
    —— 同一 collection 的索引与查询强制同 provider（元数据校验），防止向量空间错乱
  - 向量存储：Qdrant 本地模式（path 持久化，零服务）；不可用时 numpy 降级
  - Rerank：bge-reranker-v2-m3，api（newapi /v1/rerank）或本地 FlagReranker，
    失败时静默跳过（rerank 是精度增强，不是必需路径）
  - 全部配置经 .env 或环境变量读取，密钥绝不入库

依赖：numpy（必选）；qdrant-client、FlagEmbedding、requests（按所选 provider 安装）
"""
import json
import os
import numpy as np
from pathlib import Path

META_KEY = "_collection_meta"   # 存 provider/dim 的哨兵 point id


def load_env(env_path=None):
    """极简 .env 解析（KEY=VALUE 行），已存在的环境变量优先"""
    env = dict(os.environ)
    # 内网/本地服务必须直连——macOS 系统代理会把 python requests 的内网请求
    # 转发给外网代理（127.0.0.1:7890），代理够不到内网 IP → Connection refused。
    # requests 的 no_proxy 支持精确主机名匹配，逐个豁免（不影响其他域走代理）
    bypass = env.get("NO_PROXY_HOSTS",
                     "localhost,127.0.0.1")   # 内网服务地址经 .env 配置
    os.environ["NO_PROXY"] = (os.environ.get("NO_PROXY", "") + "," + bypass).lstrip(",")
    os.environ["no_proxy"] = (os.environ.get("no_proxy", "") + "," + bypass).lstrip(",")
    env["NO_PROXY"] = os.environ["NO_PROXY"]
    env["no_proxy"] = os.environ["no_proxy"]
    if env_path is None:
        env_path = Path(__file__).resolve().parent / ".env"
    if Path(env_path).exists():
        for line in Path(env_path).read_text(encoding="utf-8").split("\n"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env.setdefault(k.strip(), v.strip())
    return env


# ---------------------------------------------------------------- Embedding
class LocalBGEEmbedding:
    """BGE-m3 本地推理（FlagEmbedding），1024 维，MIT 许可"""

    def __init__(self, model_path, device=None):
        from FlagEmbedding import BGEM3FlagModel
        kwargs = {"use_fp16": True}
        if device:
            kwargs["device"] = device
        self.model = BGEM3FlagModel(model_path, **kwargs)
        self.name = f"local-bge-m3:{os.path.basename(model_path.rstrip('/'))}"
        self.dim = 1024

    def embed(self, texts, batch_size=64):
        out = self.model.encode(texts, batch_size=batch_size,
                                max_length=512)["dense_vecs"]
        return np.asarray(out, dtype=np.float32)


class ApiEmbedding:
    """newapi 网关 /v1/embeddings（OpenAI 兼容格式），qwen3-embedding 4096 维"""

    def __init__(self, base, key, model="qwen3-embedding", timeout=60):
        import requests
        self.session = requests.Session()
        self.base, self.key, self.model = base.rstrip("/"), key, model
        self.timeout = timeout
        self.name = f"api-{model}"
        self.dim = None      # 首次调用后确定

    def embed(self, texts, batch_size=32, retries=4):
        import numpy as np
        import time as _time
        vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            for attempt in range(retries):
                try:
                    r = self.session.post(
                        f"{self.base}/v1/embeddings",
                        headers={"Authorization": f"Bearer {self.key}"},
                        json={"model": self.model, "input": batch},
                        timeout=self.timeout)
                    r.raise_for_status()
                    break
                except Exception:
                    if attempt == retries - 1:
                        raise
                    _time.sleep(2 ** attempt)   # 2/4/8s 退避
            data = sorted(r.json()["data"], key=lambda d: d["index"])
            vecs.extend([d["embedding"] for d in data])
        arr = np.asarray(vecs, dtype=np.float32)
        self.dim = arr.shape[1]
        return arr


def make_embedding(provider, env):
    if provider == "local":
        return LocalBGEEmbedding(env["EMBEDDING_MODEL_PATH"])
    if provider == "api":
        return ApiEmbedding(env["NEWAPI_BASE"], env["NEWAPI_KEY"],
                            env.get("EMBEDDING_API_MODEL", "qwen3-embedding"))
    raise ValueError(f"未知 embedding provider: {provider}")


# ---------------------------------------------------------------- Rerank
class ApiRerank:
    """newapi /v1/rerank（bge-reranker-v2-m3）"""

    def __init__(self, base, key, model="bge-reranker-v2-m3", timeout=30):
        import requests
        self.session = requests.Session()
        self.base, self.key, self.model = base.rstrip("/"), key, model
        self.timeout = timeout
        self.name = f"api-{model}"

    def rerank(self, query, docs, top_k, retries=3):
        import requests
        import time as _time
        for attempt in range(retries):
            try:
                r = self.session.post(
                    f"{self.base}/v1/rerank",
                    headers={"Authorization": f"Bearer {self.key}"},
                    json={"model": self.model, "query": query, "documents": docs,
                          "top_n": top_k},
                    timeout=self.timeout)
                r.raise_for_status()
                results = sorted(r.json()["results"], key=lambda x: -x["relevance_score"])
                return [(x["index"], x["relevance_score"]) for x in results][:top_k]
            except Exception:
                if attempt == retries - 1:
                    raise
                _time.sleep(2 ** attempt)


class LocalRerank:
    """bge-reranker-v2-m3 本地推理（FlagEmbedding.FlagReranker）"""

    def __init__(self, model_path, device=None):
        from FlagEmbedding import FlagReranker
        kwargs = {"use_fp16": True}
        if device:
            kwargs["device"] = device
        self.model = FlagReranker(model_path, **kwargs)
        self.name = f"local-{os.path.basename(model_path.rstrip('/'))}"

    def rerank(self, query, docs, top_k):
        pairs = [[query, d] for d in docs]
        scores = self.model.compute_score(pairs, normalize=True)
        order = sorted(range(len(docs)), key=lambda i: -scores[i])[:top_k]
        return [(i, float(scores[i])) for i in order]


class NoRerank:
    """降级：跳过精排，保持召回顺序"""

    def rerank(self, query, docs, top_k):
        return [(i, 1.0) for i in range(min(top_k, len(docs)))]


def make_reranker(provider, env):
    if provider == "none":
        return NoRerank()
    if provider == "api":
        return ApiRerank(env["NEWAPI_BASE"], env["NEWAPI_KEY"],
                         env.get("RERANK_API_MODEL", "bge-reranker-v2-m3"))
    if provider == "local":
        return LocalRerank(env["RERANK_MODEL_PATH"])
    raise ValueError(f"未知 rerank provider: {provider}")


# ---------------------------------------------------------------- 向量存储
class QdrantStore:
    """Qdrant：服务模式（host/port，配合 docker compose）或本地模式（path 持久化），API 全同。"""

    def __init__(self, collection="tcmtodb", host=None, port=6333, path=None):
        from qdrant_client import QdrantClient
        if host:
            self.client = QdrantClient(host=host, port=int(port))
        else:
            self.client = QdrantClient(path=path)
        self.collection = collection
        self.provider_meta = {}

    def ensure_collection(self, dim, provider_name):
        from qdrant_client.models import Distance, PointStruct, VectorParams
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection not in existing:
            self.client.create_collection(
                self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
        # 哨兵点（id=0）记录 provider 元信息，查询时精确读取校验一致性
        fetched = self.client.retrieve(self.collection, ids=[0],
                                       with_payload=True, with_vectors=False)
        if fetched:
            meta = fetched[0].payload or {}
            if meta.get("_meta") != provider_name or meta.get("dim") != dim:
                raise RuntimeError(
                    f"collection 元数据不匹配：索引由 {meta.get('_meta')}"
                    f"({meta.get('dim')}维) 构建，当前 provider={provider_name}"
                    f"({dim}维)。请换 collection 或重建索引。")
        else:
            self.client.upsert(self.collection, [PointStruct(
                id=0, vector=[0.0] * dim,
                payload={"_meta": provider_name, "dim": dim})])
        self.provider_meta = {"provider": provider_name, "dim": dim}

    def count(self):
        return self.client.count(self.collection, exact=True).count

    def upsert(self, records, vectors, batch=512):
        from qdrant_client.models import PointStruct
        # id=0 是哨兵点，正文从 1 开始
        for i in range(0, len(records), batch):
            pts = [PointStruct(
                id=j + 1, vector=vec.tolist(), payload=r)
                for j, (r, vec) in enumerate(
                    zip(records[i:i + batch], vectors[i:i + batch]), start=i + 1)]
            self.client.upsert(self.collection, pts)

    def search(self, vector, top_k=50, entity_type=None):
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        flt = None
        if entity_type:
            flt = Filter(must=[FieldCondition(
                key="entity_type", match=MatchValue(value=entity_type))])
        hits = self.client.query_points(
            self.collection, query=list(vector), limit=top_k,
            query_filter=flt).points
        return [(h.payload, h.score) for h in hits
                if h.payload and not h.payload.get("_meta")]


class NumpyStore:
    """降级存储：.npz 持久化 + 暴力余弦检索（1.4 万条毫秒级）"""

    def __init__(self, path, collection="tcmtodb"):
        self.npz = Path(path).with_suffix(".npz")
        self.collection = collection
        self.records, self.matrix = [], None
        self.provider_meta = {}
        if self.npz.exists():
            z = np.load(self.npz, allow_pickle=True)
            self.records = list(z["records"])
            self.matrix = z["vectors"]
            self.provider_meta = json.loads(str(z["meta"]))

    def ensure_collection(self, dim, provider_name):
        if self.matrix is not None and (
                self.provider_meta.get("provider") != provider_name
                or self.provider_meta.get("dim") != dim):
            raise RuntimeError(
                f"索引由 {self.provider_meta.get('provider')}"
                f"({self.provider_meta.get('dim')}维) 构建，当前 provider={provider_name}({dim}维)。"
                f"删除 {self.npz} 后重建。")

    def count(self):
        return len(self.records)

    def upsert(self, records, vectors):
        self.records = list(records)
        self.matrix = vectors
        self.provider_meta = {"provider": self.provider_meta.get("provider", ""),
                              "dim": int(vectors.shape[1])}
        np.savez_compressed(self.npz, records=np.array(self.records, dtype=object),
                            vectors=self.matrix,
                            meta=np.array(json.dumps(self.provider_meta)))

    def search(self, vector, top_k=50, entity_type=None):
        if self.matrix is None or not self.records:
            return []
        v = vector / (np.linalg.norm(vector) + 1e-9)
        m = self.matrix / (np.linalg.norm(self.matrix, axis=1, keepdims=True) + 1e-9)
        scores = m @ v
        order = np.argsort(-scores)[:top_k * 2]
        out = [(self.records[i], float(scores[i])) for i in order]
        if entity_type:
            out = [(r, s) for r, s in out if r.get("entity_type") == entity_type]
        return out[:top_k]


def make_store(backend, env):
    rag_dir = Path(__file__).resolve().parent
    if backend == "qdrant":
        collection = env.get("QDRANT_COLLECTION", "tcmtodb")
        if env.get("QDRANT_HOST"):   # 服务模式（docker compose 部署）
            return QdrantStore(collection=collection,
                               host=env["QDRANT_HOST"],
                               port=int(env.get("QDRANT_PORT", 6333)))
        return QdrantStore(collection=collection,      # 本地文件模式
                           path=env.get("QDRANT_PATH", str(rag_dir / "qdrant_local")))
    if backend == "numpy":
        return NumpyStore(path=rag_dir / "vectors.npz")
    raise ValueError(f"未知存储后端: {backend}")


# ---------------------------------------------------------------- 语义检索器
class Retriever:
    """语义检索器：embedding 召回 → (可选 rerank) 精排 → 语料记录"""

    def __init__(self, corpus_path, embedding, store, reranker=None,
                 recall_k=50, top_k=5):
        self.records = [json.loads(l) for l in
                        open(corpus_path, encoding="utf-8") if l.strip()]
        self.embedding, self.store = embedding, store
        self.reranker, self.recall_k, self.top_k = reranker, recall_k, top_k

    def ensure_index(self):
        """建 collection / 灌库（幂等：已灌且 provider 一致则跳过）。
        API provider 维度在首次调用后才可知，先用探针文本确定维度。"""
        probe = self.embedding.embed(["__probe__"])
        dim = probe.shape[1]
        self.store.ensure_collection(dim, self.embedding.name)
        if hasattr(self.store, "matrix") and self.store.matrix is not None:
            return   # numpy 降级：已有索引
        try:
            current = self.store.count()
        except Exception:
            current = 0
        if current >= len(self.records):
            return   # 已灌库且数量一致
        print(f"[index] 语料 {len(self.records)} 条，库中 {current} 条 → 全量向量化"
              f"（provider={self.embedding.name}，{dim} 维）...")
        texts = [r["text"] for r in self.records]
        vecs = self.embedding.embed(texts)
        self.store.upsert(self.records, vecs)
        print(f"[index] 完成，{vecs.shape[1]} 维")

    def semantic(self, question, entity_type=None, retries=3):
        qvec = self.embedding.embed([question])[0]
        hits = []
        for attempt in range(retries):     # 存储/网络抖动重试（间歇性拒绝连接实测有效）
            try:
                hits = self.store.search(qvec, top_k=self.recall_k,
                                         entity_type=entity_type)
                break
            except Exception:
                if attempt == retries - 1:
                    raise
                import time as _time
                _time.sleep(3 * (attempt + 1))
        docs = [r for r, _ in hits]
        if self.reranker and docs:
            try:
                order = self.reranker.rerank(question, [r["text"] for r in docs],
                                             self.top_k)
                return [docs[i] for i, _ in order]
            except Exception as e:
                print(f"[warn] rerank 失败跳过精排: {e}")
        return [r for r, _ in hits[:self.top_k]]
