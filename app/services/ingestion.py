import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime


class IngestionManager:
    def __init__(self, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ingest")
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def submit(self, fn, *args, **kwargs) -> str:
        job_id = uuid.uuid4().hex[:8]
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "message": "Queued",
                "created_at": datetime.now().isoformat(),
            }
        self._executor.submit(self._run, job_id, fn, args, kwargs)
        return job_id

    def _run(self, job_id: str, fn, args, kwargs) -> None:
        self._update(job_id, status="running", message="Processing document...")
        try:
            result = fn(*args, **kwargs)
            self._update(
                job_id,
                status="completed",
                message=result.get("message", "Ingestion complete"),
                paper_id=result.get("paper_id"),
                chunks_indexed=result.get("chunks_indexed"),
            )
        except Exception as e:
            self._update(job_id, status="failed", message=str(e), chunks_indexed=0)

    def _update(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.update(**fields)

    def get_status(self, job_id: str) -> dict | None:
        with self._lock:
            job = dict(self._jobs.get(job_id, {}))
        return job or None


ingestion_manager = IngestionManager()