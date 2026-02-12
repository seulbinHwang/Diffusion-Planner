#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import fcntl  # linux only
except Exception:  # pragma: no cover
    fcntl = None


@dataclass(frozen=True)
class CopyPlanItem:
    """복사 계획의 한 항목.

    Attributes:
        src_path (str): 원본 파일 경로. shape: ()
        dst_path (str): 목적지 파일 경로. shape: ()
        rel_path (str): src_root 기준 상대 경로. shape: ()
    """
    src_path: str
    dst_path: str
    rel_path: str


def _parse_bool(v: Any) -> bool:
    """여러 형태의 입력을 bool로 바꾼다.

    Args:
        v (Any): "1", "0", "true", "false", True/False 등. shape: ()

    Returns:
        bool: 변환 결과. shape: ()
    """
    if isinstance(v, bool):
        return bool(v)
    s = str(v).strip().lower()
    return s in ("1", "true", "t", "y", "yes", "on")


def _safe_makedirs(dir_path: str) -> None:
    """폴더를 만든다(이미 있으면 넘어감).

    Args:
        dir_path (str): 만들 폴더 경로. shape: ()

    Returns:
        None
    """
    if not dir_path:
        return
    os.makedirs(dir_path, exist_ok=True)


def _read_text(path: str) -> Optional[str]:
    """텍스트 파일을 읽는다.

    Args:
        path (str): 파일 경로. shape: ()

    Returns:
        Optional[str]: 성공 시 내용 문자열, 실패 시 None.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None


def _read_json(path: str) -> Any:
    """JSON 파일을 읽는다.

    Args:
        path (str): JSON 파일 경로. shape: ()

    Returns:
        Any: json.load 결과(리스트/딕셔너리 등).
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj: Any) -> None:
    """JSON 파일을 쓴다.

    Args:
        path (str): 저장할 경로. shape: ()
        obj (Any): JSON으로 저장할 객체.

    Returns:
        None
    """
    _safe_makedirs(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def _md5_of_text(text: str) -> str:
    """문자열의 MD5 해시를 구한다.

    Args:
        text (str): 입력 문자열. shape: ()

    Returns:
        str: MD5(32글자). shape: ()
    """
    h = hashlib.md5()
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def _normalize_abs(path: str) -> str:
    """경로를 절대 경로로 정리한다.

    Args:
        path (str): 입력 경로. shape: ()

    Returns:
        str: 정리된 절대 경로. shape: ()
    """
    return os.path.abspath(os.path.expanduser(path))


def _is_under_root(abs_path: str, abs_root: str) -> bool:
    """abs_path가 abs_root 아래에 있는지 확인한다.

    Args:
        abs_path (str): 절대 경로. shape: ()
        abs_root (str): 절대 루트 경로. shape: ()

    Returns:
        bool: 아래에 있으면 True. shape: ()
    """
    try:
        rel = os.path.relpath(abs_path, abs_root)
    except Exception:
        return False
    return not rel.startswith("..")


def _collect_file_like_strings(obj: Any, *, suffix: str) -> List[str]:
    """JSON 안에서 특정 확장자를 가진 문자열을 전부 모은다.

    Args:
        obj (Any): json.load로 읽은 객체.
        suffix (str): 예: ".npz". shape: ()

    Returns:
        List[str]: 파일 경로 후보 문자열들. shape: (N,)
    """
    out: List[str] = []

    def _walk(x: Any) -> None:
        if isinstance(x, str):
            if x.endswith(suffix):
                out.append(x)
            return
        if isinstance(x, list):
            for it in x:
                _walk(it)
            return
        if isinstance(x, dict):
            # 보통 파일 경로는 value 쪽에 들어있으므로 value만 순회
            for _, v in x.items():
                _walk(v)
            return

    _walk(obj)
    return out


def _rewrite_paths_in_json(
    obj: Any,
    *,
    src_root: str,
    dst_root: str,
    suffix: str,
) -> Any:
    """JSON 구조를 유지하면서, 파일 경로 문자열만 필요하면 바꾼다.

    바꾸는 규칙(단순):
    - 문자열이 suffix(예: .npz)로 끝나고, 절대경로이며, src_root 아래에 있으면
      dst_root 아래의 대응 경로로 바꾼다.
    - 그 외는 그대로 둔다.
      (상대경로면 그대로 두어도, 학습 코드가 train_set(root)와 합쳐서 찾게 됨)

    Args:
        obj (Any): 원본 JSON 객체.
        src_root (str): 원본 데이터 루트(절대경로 권장). shape: ()
        dst_root (str): 로컬 데이터 루트(절대경로 권장). shape: ()
        suffix (str): 파일 확장자. shape: ()

    Returns:
        Any: 바뀐 JSON 객체(구조는 동일).
    """
    if isinstance(obj, str):
        s = obj
        if s.endswith(suffix) and os.path.isabs(s):
            abs_s = _normalize_abs(s)
            abs_src = _normalize_abs(src_root)
            abs_dst = _normalize_abs(dst_root)
            if _is_under_root(abs_s, abs_src):
                rel = os.path.relpath(abs_s, abs_src)
                return os.path.join(abs_dst, rel)
        return s

    if isinstance(obj, list):
        return [_rewrite_paths_in_json(it, src_root=src_root, dst_root=dst_root, suffix=suffix) for it in obj]

    if isinstance(obj, dict):
        new_d: Dict[Any, Any] = {}
        for k, v in obj.items():
            new_d[k] = _rewrite_paths_in_json(v, src_root=src_root, dst_root=dst_root, suffix=suffix)
        return new_d

    return obj


def _build_copy_plan(
    *,
    src_root: str,
    dst_root: str,
    file_entries: Sequence[str],
) -> List[CopyPlanItem]:
    """복사 계획(원본→목적지 경로 목록)을 만든다.

    Args:
        src_root (str): 원본 데이터 루트. shape: ()
        dst_root (str): 로컬 데이터 루트. shape: ()
        file_entries (Sequence[str]): JSON에서 찾은 파일 경로/이름 문자열들. shape: (N,)

    Returns:
        List[CopyPlanItem]: 복사 계획 목록. shape: (N_unique,)

    Raises:
        FileNotFoundError: 원본 파일이 실제로 없으면.
        ValueError: 경로 해석이 불가능한 항목이 있으면.
    """
    abs_src_root = _normalize_abs(src_root)
    abs_dst_root = _normalize_abs(dst_root)

    seen_rel: set[str] = set()
    plan: List[CopyPlanItem] = []

    for entry in file_entries:
        if not isinstance(entry, str) or not entry:
            continue

        if os.path.isabs(entry):
            abs_entry = _normalize_abs(entry)
            if not _is_under_root(abs_entry, abs_src_root):
                # src_root 밖의 절대경로는 여기서 안전하게 처리하기 어렵다.
                raise ValueError(
                    f"train_set_list 안의 절대경로가 src_root 밖에 있습니다: entry='{entry}', src_root='{abs_src_root}'"
                )
            rel = os.path.relpath(abs_entry, abs_src_root)
            src_path = abs_entry
        else:
            rel = entry
            src_path = os.path.join(abs_src_root, rel)

        rel_norm = os.path.normpath(rel)
        if rel_norm in seen_rel:
            continue
        seen_rel.add(rel_norm)

        dst_path = os.path.join(abs_dst_root, rel_norm)

        if not os.path.isfile(src_path):
            raise FileNotFoundError(f"원본 파일을 찾을 수 없습니다: {src_path}")

        plan.append(CopyPlanItem(src_path=src_path, dst_path=dst_path, rel_path=rel_norm))

    return plan


def _copy_one_file_atomic(src_path: str, dst_path: str) -> None:
    """파일을 '중간에 깨질 확률을 줄이는 방식'으로 복사한다.

    방식:
    - 목적지와 같은 폴더에 임시 파일로 먼저 복사
    - 마지막에 이름을 교체(os.replace)

    Args:
        src_path (str): 원본 파일 경로. shape: ()
        dst_path (str): 목적지 파일 경로. shape: ()

    Returns:
        None
    """
    dst_dir = os.path.dirname(dst_path)
    _safe_makedirs(dst_dir)

    tmp_path = f"{dst_path}.tmp.{os.getpid()}"
    shutil.copyfile(src_path, tmp_path)
    os.replace(tmp_path, dst_path)


def _filter_copy_plan_by_existing_files(plan: Sequence[CopyPlanItem]) -> List[CopyPlanItem]:
    """이미 목적지에 같은 크기의 파일이 있으면 복사 계획에서 제외한다.

    Args:
        plan (Sequence[CopyPlanItem]): 복사 계획. shape: (N,)

    Returns:
        List[CopyPlanItem]: 실제로 복사해야 하는 계획. shape: (M,)
    """
    todo: List[CopyPlanItem] = []
    for it in plan:
        if os.path.isfile(it.dst_path):
            try:
                if os.path.getsize(it.dst_path) == os.path.getsize(it.src_path):
                    continue
            except Exception:
                # 크기 확인 실패 시에는 안전하게 다시 복사
                pass
        todo.append(it)
    return todo


def _copy_files_parallel(
    plan: Sequence[CopyPlanItem],
    *,
    num_workers: int,
    log_every: int,
) -> None:
    """복사 계획을 여러 작업자로 나눠서 복사한다.

    Args:
        plan (Sequence[CopyPlanItem]): 복사 계획. shape: (N,)
        num_workers (int): 동시에 돌릴 작업자 수. shape: ()
        log_every (int): 진행 로그를 몇 개마다 찍을지. shape: ()

    Returns:
        None
    """
    total = int(len(plan))
    if total <= 0:
        return

    n_workers = int(max(1, num_workers))
    log_every_i = int(max(1, log_every))

    start = time.time()

    def _worker(it: CopyPlanItem) -> None:
        _copy_one_file_atomic(it.src_path, it.dst_path)

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        # executor.map을 쓰면 큰 목록에서도 메모리를 과하게 쓰지 않게 동작하는 편이다.
        for idx, _ in enumerate(ex.map(_worker, plan), start=1):
            if (idx % log_every_i == 0) or (idx == total):
                elapsed = time.time() - start
                print(f"[LOCAL_DATA] copy {idx}/{total} done (elapsed {elapsed:.1f}s)", flush=True)


def _read_marker(marker_path: str) -> Optional[Dict[str, Any]]:
    """마커 파일(.json)을 읽는다.

    Args:
        marker_path (str): 마커 파일 경로. shape: ()

    Returns:
        Optional[Dict[str, Any]]: 성공 시 dict, 실패 시 None.
    """
    if not os.path.isfile(marker_path):
        return None
    try:
        with open(marker_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return obj
        return None
    except Exception:
        return None


def _write_marker(marker_path: str, payload: Dict[str, Any]) -> None:
    """마커 파일(.json)을 쓴다.

    Args:
        marker_path (str): 마커 파일 경로. shape: ()
        payload (Dict[str, Any]): 저장할 내용. shape: ()

    Returns:
        None
    """
    _safe_makedirs(os.path.dirname(marker_path))
    tmp_path = f"{marker_path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
    os.replace(tmp_path, marker_path)


def _acquire_lock(lock_path: str) -> Optional[Any]:
    """간단한 파일 잠금을 잡는다(가능한 환경에서만).

    Args:
        lock_path (str): 잠금 파일 경로. shape: ()

    Returns:
        Optional[Any]:
            - 성공: 열려있는 파일 객체(해제 시 close 필요)
            - 실패/미지원: None
    """
    if fcntl is None:
        return None

    _safe_makedirs(os.path.dirname(lock_path))
    f = open(lock_path, "a", encoding="utf-8")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        return f
    except Exception:
        try:
            f.close()
        except Exception:
            pass
        return None


def _release_lock(lock_file_obj: Optional[Any]) -> None:
    """파일 잠금을 해제한다.

    Args:
        lock_file_obj (Optional[Any]): _acquire_lock에서 받은 파일 객체.

    Returns:
        None
    """
    if lock_file_obj is None:
        return
    try:
        if fcntl is not None:
            fcntl.flock(lock_file_obj.fileno(), fcntl.LOCK_UN)
    except Exception:
        pass
    try:
        lock_file_obj.close()
    except Exception:
        pass


def prepare_local_train_set(
    *,
    src_root: str,
    src_list_path: str,
    dst_root: str,
    dst_list_path: str,
    force_rebuild: bool,
    suffix: str,
    num_workers: int,
) -> Tuple[str, str]:
    """원본 학습 데이터를 로컬 경로로 복사하고, 로컬용 list 파일을 만든다.

    핵심 규칙:
    - 이미 한 번 "완료"된 상태(마커 파일 존재 + list 해시가 동일)면 복사를 건너뜀
    - force_rebuild=True면 항상 다시 복사(로컬 폴더를 지우고 새로 만듦)
    - 복사는 src_list_path에 들어있는 파일들만 대상으로 함

    Args:
        src_root (str): 원본 데이터 루트(예: /mnt/.../processed). shape: ()
        src_list_path (str): 원본 list(json) 파일 경로. shape: ()
        dst_root (str): 로컬 데이터 루트(예: /workspace/local_shards_v1). shape: ()
        dst_list_path (str): 로컬 list(json) 파일 경로. shape: ()
        force_rebuild (bool): True면 항상 다시 복사. shape: ()
        suffix (str): 찾을 파일 확장자(기본 .npz). shape: ()
        num_workers (int): 복사 작업자 수. shape: ()

    Returns:
        Tuple[str, str]:
            - dst_root(절대경로)
            - dst_list_path(절대경로)
    """
    abs_src_root = _normalize_abs(src_root)
    abs_src_list = _normalize_abs(src_list_path)
    abs_dst_root = _normalize_abs(dst_root)
    abs_dst_list = _normalize_abs(dst_list_path)

    if not os.path.isdir(abs_src_root):
        raise FileNotFoundError(f"src_root 폴더가 없습니다: {abs_src_root}")
    if not os.path.isfile(abs_src_list):
        raise FileNotFoundError(f"src_list_path 파일이 없습니다: {abs_src_list}")

    # 마커/락 경로
    marker_path = os.path.join(abs_dst_root, ".local_train_set_marker.json")
    lock_path = os.path.join(abs_dst_root, ".local_train_set_copy.lock")

    lock_obj = _acquire_lock(lock_path)
    try:
        src_list_text = _read_text(abs_src_list)
        if src_list_text is None:
            raise RuntimeError(f"src_list_path를 읽지 못했습니다: {abs_src_list}")
        src_list_md5 = _md5_of_text(src_list_text)

        # force_rebuild면 로컬 폴더를 지우고 새로 시작
        if force_rebuild:
            if os.path.isdir(abs_dst_root):
                print(f"[LOCAL_DATA] force_rebuild=True -> remove dst_root: {abs_dst_root}", flush=True)
                shutil.rmtree(abs_dst_root, ignore_errors=True)
            _safe_makedirs(abs_dst_root)

        # 이미 준비 완료면 스킵
        marker = _read_marker(marker_path)
        if (not force_rebuild) and marker is not None:
            if (marker.get("src_list_md5") == src_list_md5) and os.path.isfile(abs_dst_list):
                print(f"[LOCAL_DATA] already prepared. skip copy. dst_root={abs_dst_root}", flush=True)
                return abs_dst_root, abs_dst_list

        # list JSON 읽기
        src_list_json = _read_json(abs_src_list)

        # 복사할 파일 목록 뽑기
        file_entries = _collect_file_like_strings(src_list_json, suffix=suffix)
        if not file_entries:
            raise ValueError(
                f"src_list_path 안에서 '{suffix}'로 끝나는 파일 항목을 찾지 못했습니다: {abs_src_list}"
            )

        # 복사 계획 만들기
        plan_all = _build_copy_plan(
            src_root=abs_src_root,
            dst_root=abs_dst_root,
            file_entries=file_entries,
        )

        # 이미 복사된 파일은 크기 기준으로 건너뛰기(중간에 끊긴 경우 재시도에 유리)
        plan_todo = _filter_copy_plan_by_existing_files(plan_all)

        print(
            f"[LOCAL_DATA] src_root={abs_src_root}\n"
            f"[LOCAL_DATA] dst_root={abs_dst_root}\n"
            f"[LOCAL_DATA] files(total)={len(plan_all)}, files(to_copy)={len(plan_todo)}",
            flush=True,
        )

        if plan_todo:
            # 너무 자주 찍지 않게 적당히 설정
            log_every = 200
            _copy_files_parallel(
                plan=plan_todo,
                num_workers=int(max(1, num_workers)),
                log_every=log_every,
            )

        # 로컬용 list 만들기(필요하면 절대경로 항목만 src->dst로 바꿈)
        dst_list_json = _rewrite_paths_in_json(
            src_list_json,
            src_root=abs_src_root,
            dst_root=abs_dst_root,
            suffix=suffix,
        )
        _write_json(abs_dst_list, dst_list_json)

        # 마커 기록
        payload: Dict[str, Any] = {
            "created_at_unix": int(time.time()),
            "src_root": abs_src_root,
            "dst_root": abs_dst_root,
            "src_list_path": abs_src_list,
            "dst_list_path": abs_dst_list,
            "src_list_md5": src_list_md5,
            "suffix": str(suffix),
            "num_files": int(len(plan_all)),
        }
        _write_marker(marker_path, payload)

        print(f"[LOCAL_DATA] done. dst_list_path={abs_dst_list}", flush=True)
        return abs_dst_root, abs_dst_list
    finally:
        _release_lock(lock_obj)


def _build_arg_parser() -> argparse.ArgumentParser:
    """CLI 인자 파서를 만든다."""
    p = argparse.ArgumentParser(description="Copy train .npz files to local path (one-time cache).")
    p.add_argument("--src_root", type=str, required=True, help="원본 데이터 루트 경로")
    p.add_argument("--src_list", type=str, required=True, help="원본 train_set_list(json) 경로")
    p.add_argument("--dst_root", type=str, required=True, help="로컬 데이터 루트 경로")
    p.add_argument("--dst_list", type=str, required=True, help="로컬 train_set_list(json) 저장 경로")
    p.add_argument("--force_rebuild", type=str, default="false", help="true면 항상 다시 복사")
    p.add_argument("--suffix", type=str, default=".npz", help="복사 대상 파일 확장자(기본 .npz)")
    p.add_argument("--num_workers", type=int, default=8, help="복사 작업자 수(기본 8)")
    return p


def main() -> None:
    """스크립트 진입점."""
    args = _build_arg_parser().parse_args()
    prepare_local_train_set(
        src_root=str(args.src_root),
        src_list_path=str(args.src_list),
        dst_root=str(args.dst_root),
        dst_list_path=str(args.dst_list),
        force_rebuild=_parse_bool(args.force_rebuild),
        suffix=str(args.suffix),
        num_workers=int(args.num_workers),
    )


if __name__ == "__main__":
    main()
