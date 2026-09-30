"""벡터스토어 클라이언트 동시 생성 테스트 (LangGraph 병렬 노드 재현)."""
import threading

from rag import ingest


def test_concurrent_get_collection_creates_single_client(monkeypatch, tmp_path):
    monkeypatch.setattr(ingest, "DATA_DIR", tmp_path)
    ingest._create_client.cache_clear()
    errors, clients = [], []
    barrier = threading.Barrier(8)

    def work():
        barrier.wait()  # 8개 스레드가 동시에 첫 호출
        try:
            ingest.get_collection("tech_docs").count()
            clients.append(ingest._client())
        except Exception as error:
            errors.append(repr(error))

    threads = [threading.Thread(target=work) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    ingest._create_client.cache_clear()  # 다른 테스트가 tmp 경로 클라이언트를 쓰지 않도록

    assert errors == []
    assert len({id(client) for client in clients}) == 1
