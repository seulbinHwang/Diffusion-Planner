#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python gpu_node_watcher.py --init


python gpu_node_watcher.py

python gpu_node_watcher.py --headful --once

"""
"""
GPU 노드 감시기

- 지정한 Grafana 링크를 자동으로 열어 표를 읽습니다.
- 각 노드의 '할당된 총 GPU 수'를 확인합니다.
- '할당된 총 GPU 수'가 기준값(기본 2) 이하인 노드가 1대라도 있으면 이메일을 보냅니다.
- 같은 상태가 계속될 때 이메일이 너무 많이 가지 않도록, "상태가 처음 바뀌는 순간"에만 보냅니다.

필수 설치:
    pip install playwright
    python -m playwright install chromium

선택 설치(비밀번호를 OS에 저장하고 싶을 때):
    pip install keyring
"""

from __future__ import annotations

import argparse
import getpass
import json
import logging
import os
import re
import smtplib
import ssl
import time
from dataclasses import asdict, dataclass
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Optional

from playwright.sync_api import Page, sync_playwright


LOGGER = logging.getLogger("gpu_node_watcher")


# =========================
# 데이터 구조
# =========================

@dataclass(frozen=True)
class EmailSettings:
    """이메일 발송 설정값 묶음.

    Args:
        smtp_host (str): 메일을 보내는 서버 주소(예: smtp.gmail.com).
        smtp_port (int): 메일을 보내는 서버 포트(예: 587).
        smtp_username (str): 메일 서버 로그인에 쓰는 아이디(대부분 이메일 주소).
        from_email (str): 받는 사람이 보게 될 '보낸 사람' 주소.
        to_email (str): 알림을 받을 이메일 주소.
        use_starttls (bool): 안전한 연결로 바꿔서 보낼지 여부(대부분 True 권장).
    """
    smtp_host: str
    smtp_port: int
    smtp_username: str
    from_email: str
    to_email: str
    use_starttls: bool


@dataclass(frozen=True)
class WatchSettings:
    """감시(확인) 설정값 묶음.

    Args:
        grafana_url (str): 확인할 Grafana 링크.
        check_interval_minutes (int): 몇 분마다 확인할지(예: 10).
        allocated_gpu_threshold (int): '할당된 총 GPU 수'가 이 값 이하이면 알림 조건 성립.
        table_wait_seconds (int): 표가 뜰 때까지 최대 몇 초 기다릴지.
        browser_profile_dir (str): 브라우저 로그인/쿠키를 저장할 폴더 경로.
        headless (bool): 화면을 띄우지 않고 실행할지 여부(True면 화면 없음).
    """
    grafana_url: str
    check_interval_minutes: int
    allocated_gpu_threshold: int
    table_wait_seconds: int
    browser_profile_dir: str
    headless: bool


@dataclass(frozen=True)
class AppConfig:
    """프로그램 전체 설정값 묶음.

    Args:
        email (EmailSettings): 이메일 설정.
        watch (WatchSettings): 감시 설정.
        password_storage (str): 비밀번호를 저장하는 방식. "keyring" 또는 "file".
    """
    email: EmailSettings
    watch: WatchSettings
    password_storage: str  # "keyring" or "file"


@dataclass
class AlertState:
    """알림 중복 발송을 막기 위한 상태값.

    Args:
        alert_active (bool): 직전 확인에서 '조건 만족 노드가 있었는지' 여부.
    """
    alert_active: bool = False


# =========================
# 파일/설정 입출력
# =========================

def _get_app_dir() -> Path:
    """이 프로그램이 쓰는 폴더 경로를 돌려줍니다.

    Returns:
        Path: 예) ~/.gpu_node_watcher
    """
    return Path.home() / ".gpu_node_watcher"


def _get_config_path() -> Path:
    """설정 파일 경로를 돌려줍니다.

    Returns:
        Path: 예) ~/.gpu_node_watcher/config.json
    """
    return _get_app_dir() / "config.json"


def _get_state_path() -> Path:
    """상태 파일 경로를 돌려줍니다.

    Returns:
        Path: 예) ~/.gpu_node_watcher/state.json
    """
    return _get_app_dir() / "state.json"


def _get_password_file_path() -> Path:
    """비밀번호를 파일로 저장할 때 쓰는 경로를 돌려줍니다.

    Returns:
        Path: 예) ~/.gpu_node_watcher/email_password.txt
    """
    return _get_app_dir() / "email_password.txt"


def _ensure_app_dir_exists(app_dir: Path) -> None:
    """프로그램 폴더가 없으면 생성합니다.

    Args:
        app_dir (Path): 프로그램 폴더 경로.
    """
    app_dir.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """JSON 파일로 저장합니다.

    Args:
        path (Path): 저장할 파일 경로.
        data (dict[str, Any]): 저장할 내용.
    """
    _ensure_app_dir_exists(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    """JSON 파일을 읽습니다.

    Args:
        path (Path): 읽을 파일 경로.

    Returns:
        dict[str, Any]: 읽어온 내용.

    Raises:
        FileNotFoundError: 파일이 없을 때.
        json.JSONDecodeError: JSON 형식이 깨졌을 때.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def _set_private_permissions(path: Path) -> None:
    """파일 권한을 '나만 읽기/쓰기'로 최대한 제한합니다(가능한 환경에서만 적용).

    Args:
        path (Path): 권한을 제한할 파일 경로.
    """
    try:
        os.chmod(path, 0o600)
    except Exception:
        # 환경에 따라 실패할 수 있어도, 프로그램 실행 자체는 계속되게 둡니다.
        pass


def _prompt_text(prompt: str, default: Optional[str] = None) -> str:
    """사용자에게 문자열 입력을 받습니다.

    Args:
        prompt (str): 사용자에게 보여줄 질문 문장.
        default (Optional[str]): 엔터만 치면 사용할 기본값.

    Returns:
        str: 입력된 값(또는 기본값).
    """
    if default is None:
        value = input(f"{prompt}: ").strip()
        return value
    value = input(f"{prompt} (기본값: {default}): ").strip()
    return value if value else default


def _prompt_int(prompt: str, default: int) -> int:
    """사용자에게 정수 입력을 받습니다.

    Args:
        prompt (str): 사용자에게 보여줄 질문 문장.
        default (int): 엔터만 치면 사용할 기본값.

    Returns:
        int: 입력된 값(또는 기본값).
    """
    while True:
        raw = input(f"{prompt} (기본값: {default}): ").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            print("숫자로 입력해줘.")


def _load_or_create_config(config_path: Path) -> AppConfig:
    """설정을 읽거나, 없으면 사용자 입력으로 새로 만듭니다.

    이 프로그램은 최초 1회만 필요한 값을 물어보고 파일로 저장합니다.
    다음부터는 저장된 값을 그대로 씁니다.

    Args:
        config_path (Path): 설정 파일 경로.

    Returns:
        AppConfig: 설정값 묶음.
    """
    if config_path.exists():
        raw = _read_json(config_path)
        email = EmailSettings(**raw["email"])
        watch = WatchSettings(**raw["watch"])
        password_storage = raw.get("password_storage", "keyring")
        return AppConfig(email=email, watch=watch, password_storage=password_storage)

    print("\n[최초 1회 설정] 아래 질문에 답하면, 다음부터는 다시 묻지 않습니다.\n")

    grafana_url = _prompt_text("확인할 Grafana 링크(URL)")
    check_interval_minutes = _prompt_int("몇 분마다 확인할까요?", default=10)
    allocated_gpu_threshold = _prompt_int("할당된 총 GPU 수가 몇 이하면 알릴까요?", default=2)
    table_wait_seconds = _prompt_int("표가 뜰 때까지 최대 몇 초 기다릴까요?", default=60)

    default_profile_dir = str(_get_app_dir() / "browser_profile")
    browser_profile_dir = _prompt_text("브라우저 정보(로그인/쿠키) 저장 폴더", default=default_profile_dir)

    print("\n[이메일 설정] 메일을 보내기 위한 서버 정보가 필요합니다.")
    smtp_host = _prompt_text("SMTP 서버 주소", default="smtp.gmail.com")
    smtp_port = _prompt_int("SMTP 포트", default=587)
    smtp_username = _prompt_text("SMTP 로그인 아이디(대부분 이메일 주소)")
    from_email = _prompt_text("보낸 사람 이메일", default=smtp_username)
    to_email = _prompt_text("받는 사람 이메일", default=smtp_username)
    use_starttls_raw = _prompt_text("STARTTLS 사용? (y/n)", default="y").lower().strip()
    use_starttls = use_starttls_raw in ("y", "yes")

    password_storage_raw = _prompt_text("비밀번호 저장 방식: keyring(권장) / file", default="keyring").lower().strip()
    password_storage = password_storage_raw if password_storage_raw in ("keyring", "file") else "keyring"

    cfg = AppConfig(
        email=EmailSettings(
            smtp_host=smtp_host,
            smtp_port=smtp_port,
            smtp_username=smtp_username,
            from_email=from_email,
            to_email=to_email,
            use_starttls=use_starttls,
        ),
        watch=WatchSettings(
            grafana_url=grafana_url,
            check_interval_minutes=check_interval_minutes,
            allocated_gpu_threshold=allocated_gpu_threshold,
            table_wait_seconds=table_wait_seconds,
            browser_profile_dir=browser_profile_dir,
            headless=True,
        ),
        password_storage=password_storage,
    )

    _write_json(config_path, asdict(cfg))
    print(f"\n설정 저장 완료: {config_path}\n")
    return cfg


def _load_state(state_path: Path) -> AlertState:
    """알림 상태를 읽습니다.

    Args:
        state_path (Path): 상태 파일 경로.

    Returns:
        AlertState: 상태값(없으면 기본값).
    """
    if not state_path.exists():
        return AlertState(alert_active=False)
    try:
        raw = _read_json(state_path)
        return AlertState(alert_active=bool(raw.get("alert_active", False)))
    except Exception:
        return AlertState(alert_active=False)


def _save_state(state_path: Path, state: AlertState) -> None:
    """알림 상태를 저장합니다.

    Args:
        state_path (Path): 상태 파일 경로.
        state (AlertState): 저장할 상태값.
    """
    _write_json(state_path, {"alert_active": state.alert_active})


# =========================
# 비밀번호 저장/읽기
# =========================

def _try_keyring_set(service: str, username: str, password: str) -> bool:
    """OS에 비밀번호를 저장하려고 시도합니다.

    이 방식은 보통 파일에 직접 적는 것보다 안전합니다.
    다만 환경에 따라 keyring이 없거나, 저장 기능이 막혀 있을 수 있습니다.

    Args:
        service (str): 저장할 서비스 이름(프로그램 이름 같은 값).
        username (str): 계정 이름(여기서는 이메일).
        password (str): 비밀번호.

    Returns:
        bool: 저장에 성공하면 True, 실패하면 False.
    """
    try:
        import keyring  # type: ignore
        keyring.set_password(service, username, password)
        return True
    except Exception:
        return False


def _try_keyring_get(service: str, username: str) -> Optional[str]:
    """OS에서 비밀번호를 읽으려고 시도합니다.

    Args:
        service (str): 저장된 서비스 이름.
        username (str): 계정 이름.

    Returns:
        Optional[str]: 비밀번호(있으면 문자열), 없으면 None.
    """
    try:
        import keyring  # type: ignore
        return keyring.get_password(service, username)
    except Exception:
        return None


def _save_password_to_file(path: Path, password: str) -> None:
    """비밀번호를 파일에 저장합니다.

    주의: 이 방식은 환경에 따라 위험할 수 있습니다.
    파일 권한을 최대한 줄이려고 시도하지만, 완벽하다고 보장할 수는 없습니다.

    Args:
        path (Path): 저장할 파일 경로.
        password (str): 비밀번호.
    """
    _ensure_app_dir_exists(path.parent)
    path.write_text(password, encoding="utf-8")
    _set_private_permissions(path)


def _load_password_from_file(path: Path) -> Optional[str]:
    """비밀번호를 파일에서 읽습니다.

    Args:
        path (Path): 읽을 파일 경로.

    Returns:
        Optional[str]: 비밀번호(있으면 문자열), 없으면 None.
    """
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return None


def _get_or_ask_email_password(config: AppConfig) -> str:
    """이메일 비밀번호를 가져옵니다. 없으면 1번만 입력받아 저장합니다.

    Args:
        config (AppConfig): 전체 설정값.

    Returns:
        str: 이메일 비밀번호.
    """
    service_name = "gpu_node_watcher_email"
    username = config.email.smtp_username

    if config.password_storage == "keyring":
        saved = _try_keyring_get(service_name, username)
        if saved:
            return saved

        pw = getpass.getpass("SMTP 비밀번호(또는 앱 비밀번호)를 입력해줘(최초 1회): ")
        ok = _try_keyring_set(service_name, username, pw)
        if ok:
            print("비밀번호를 OS 저장소에 저장했습니다. 다음부터 다시 묻지 않습니다.\n")
            return pw

        print("OS 저장소에 저장하지 못했습니다. 대신 파일 저장 방식으로 진행합니다.\n")
        _save_password_to_file(_get_password_file_path(), pw)
        return pw

    # file 저장 방식
    saved_file = _load_password_from_file(_get_password_file_path())
    if saved_file:
        return saved_file

    pw = getpass.getpass("SMTP 비밀번호(또는 앱 비밀번호)를 입력해줘(최초 1회): ")
    _save_password_to_file(_get_password_file_path(), pw)
    print(f"비밀번호를 파일에 저장했습니다: {_get_password_file_path()}\n")
    return pw


# =========================
# 이메일 발송
# =========================

def _send_email(settings: EmailSettings, smtp_password: str, subject: str, body: str) -> None:
    """이메일을 1통 보냅니다.

    Args:
        settings (EmailSettings): 이메일 발송 설정.
        smtp_password (str): SMTP 비밀번호(또는 앱 비밀번호).
        subject (str): 이메일 제목.
        body (str): 이메일 내용(본문).

    Raises:
        RuntimeError: 이메일 전송에 실패했을 때.
    """
    msg = EmailMessage()
    msg["From"] = settings.from_email
    msg["To"] = settings.to_email
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        if settings.use_starttls:
            context = ssl.create_default_context()
            with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30) as server:
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()
                server.login(settings.smtp_username, smtp_password)
                server.send_message(msg)
        else:
            # 일부 환경은 SSL 포트를 바로 쓰기도 합니다. 필요하면 포트를 맞춰야 합니다.
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port, context=context, timeout=30) as server:
                server.login(settings.smtp_username, smtp_password)
                server.send_message(msg)
    except Exception as e:
        raise RuntimeError(f"이메일 전송 실패: {e}") from e


# =========================
# 웹페이지에서 표 읽기
# =========================

def _parse_first_int(text: str) -> Optional[int]:
    """문자열에서 처음 나타나는 정수를 찾아서 돌려줍니다.

    Args:
        text (str): 예) "8", "8 / 8", "87.5%", "GPU 4개"

    Returns:
        Optional[int]: 찾으면 정수, 못 찾으면 None.
    """
    m = re.search(r"-?\d+", text)
    if not m:
        return None
    try:
        return int(m.group(0))
    except ValueError:
        return None


def _find_header_index(headers: list[str], target: str) -> Optional[int]:
    """열 이름 목록에서 목표 열의 위치를 찾습니다.

    Args:
        headers (list[str]): 길이 C의 열 이름 목록.
        target (str): 찾고 싶은 열 이름(완전 일치가 아니어도 포함이면 성공).

    Returns:
        Optional[int]: 열 위치(0부터 시작). 못 찾으면 None.
    """
    for i, h in enumerate(headers):
        if target in h:
            return i
    return None


def _find_header_index_any(headers: list[str], candidates: list[str]) -> Optional[int]:
    """여러 후보 중 하나라도 맞는 열의 위치를 찾습니다.

    Args:
        headers (list[str]): 길이 C의 열 이름 목록.
        candidates (list[str]): 길이 K의 후보 열 이름 목록.

    Returns:
        Optional[int]: 열 위치(0부터 시작). 못 찾으면 None.
    """
    for cand in candidates:
        idx = _find_header_index(headers, cand)
        if idx is not None:
            return idx
    return None


def _extract_table_using_roles(page: Page) -> list[tuple[str, int]]:
    """화면에서 표를 읽어 (노드, 할당 GPU 수) 목록을 만듭니다.

    이 함수는 화면의 표에서:
    - 열 이름(예: Node, 할당된 총 GPU 수)을 먼저 찾고,
    - 각 행에서 해당 열의 값을 읽습니다.

    Returns:
        list[tuple[str, int]]: 길이 N의 목록.
            각 원소는 (노드이름, 할당된총GPU수) 입니다.
    """
    header_locator = page.locator("[role='columnheader']")
    header_count = header_locator.count()
    if header_count <= 0:
        raise ValueError("표의 열 이름을 찾지 못했습니다(role='columnheader' 없음).")

    headers: list[str] = []
    # headers shape: (C,)
    for i in range(header_count):
        headers.append(header_locator.nth(i).inner_text().strip())

    allocated_idx = _find_header_index(headers, "할당된 총 GPU 수")
    node_idx = _find_header_index_any(headers, ["Node", "노드"])

    if allocated_idx is None or node_idx is None:
        raise ValueError(f"필요한 열을 찾지 못했습니다. headers={headers}")

    row_locator = page.locator("[role='row']")
    row_count = row_locator.count()
    if row_count <= 0:
        raise ValueError("표의 행을 찾지 못했습니다(role='row' 없음).")

    results: list[tuple[str, int]] = []
    # results shape: (N, 2)
    for r in range(row_count):
        row = row_locator.nth(r)
        cell_locator = row.locator("[role='cell']")
        cell_count = cell_locator.count()
        if cell_count <= 0:
            continue
        if cell_count <= max(allocated_idx, node_idx):
            continue

        node_cell = cell_locator.nth(node_idx)
        alloc_cell = cell_locator.nth(allocated_idx)

        node_text = (node_cell.get_attribute("title") or node_cell.inner_text() or "").strip()
        alloc_text = (alloc_cell.get_attribute("title") or alloc_cell.inner_text() or "").strip()
        alloc_val = _parse_first_int(alloc_text)

        if not node_text or alloc_val is None:
            continue

        results.append((node_text, alloc_val))

    if not results:
        raise ValueError("표에서 데이터를 읽었지만 결과가 비었습니다(행 파싱 실패).")

    return results


def _extract_table_using_html_table(page: Page) -> list[tuple[str, int]]:
    """<table> 태그 기반 표가 있을 때 (노드, 할당 GPU 수) 목록을 읽습니다.

    Grafana 화면은 환경에 따라 <table> 태그가 아닐 수도 있습니다.
    그래서 이 함수는 '대체 방법'입니다.

    Returns:
        list[tuple[str, int]]: 길이 N의 목록.
            각 원소는 (노드이름, 할당된총GPU수) 입니다.
    """
    table = page.locator("table").first
    if table.count() == 0:
        raise ValueError("<table> 태그를 찾지 못했습니다.")

    headers = [h.strip() for h in table.locator("thead tr th").all_inner_texts()]
    allocated_idx = _find_header_index(headers, "할당된 총 GPU 수")
    node_idx = _find_header_index_any(headers, ["Node", "노드"])
    if allocated_idx is None or node_idx is None:
        raise ValueError(f"<table> 기반에서 필요한 열을 찾지 못했습니다. headers={headers}")

    rows = table.locator("tbody tr")
    row_count = rows.count()
    if row_count <= 0:
        raise ValueError("<table> 기반에서 행을 찾지 못했습니다.")

    results: list[tuple[str, int]] = []
    # results shape: (N, 2)
    for i in range(row_count):
        row = rows.nth(i)
        cells = row.locator("td")
        cell_count = cells.count()
        if cell_count <= max(allocated_idx, node_idx):
            continue

        node_text = (cells.nth(node_idx).get_attribute("title") or cells.nth(node_idx).inner_text() or "").strip()
        alloc_text = (cells.nth(allocated_idx).get_attribute("title") or cells.nth(allocated_idx).inner_text() or "").strip()
        alloc_val = _parse_first_int(alloc_text)

        if not node_text or alloc_val is None:
            continue
        results.append((node_text, alloc_val))

    if not results:
        raise ValueError("<table> 기반에서 결과가 비었습니다.")
    return results


def fetch_node_allocated_gpu_counts(
    url: str,
    browser_profile_dir: Path,
    headless: bool,
    table_wait_seconds: int,
) -> list[tuple[str, int]]:
    """Grafana 링크를 열어서 노드별 '할당된 총 GPU 수' 목록을 읽어옵니다.

    이 함수는:
    1) 링크를 자동으로 열고,
    2) '할당된 총 GPU 수'라는 글자가 화면에 나타날 때까지 기다리고,
    3) 표에서 (노드, 할당GPU수)를 읽어서 돌려줍니다.

    Args:
        url (str): Grafana 링크.
        browser_profile_dir (Path): 로그인/쿠키를 저장할 폴더.
        headless (bool): True면 화면 없이 실행, False면 화면을 띄움.
        table_wait_seconds (int): 표가 뜰 때까지 최대 대기 시간(초).

    Returns:
        list[tuple[str, int]]: 길이 N의 목록.
            각 원소는 (노드이름, 할당된총GPU수) 입니다.

    Raises:
        RuntimeError: 페이지 열기/표 읽기에 실패했을 때.
    """
    browser_profile_dir.mkdir(parents=True, exist_ok=True)

    try:
        with sync_playwright() as p:
            context = p.chromium.launch_persistent_context(
                user_data_dir=str(browser_profile_dir),
                headless=headless,
            )
            page = context.new_page()
            page.set_default_timeout(table_wait_seconds * 1000)

            page.goto(url, wait_until="domcontentloaded")
            # 표의 핵심 열이 화면에 뜰 때까지 기다립니다.
            page.wait_for_selector("text=할당된 총 GPU 수")

            # 표가 늦게 채워지는 경우를 대비해 아주 짧게 더 기다립니다.
            page.wait_for_timeout(1500)

            try:
                data = _extract_table_using_roles(page)
            except Exception as first_err:
                LOGGER.warning("role 기반 읽기 실패: %s", first_err)
                data = _extract_table_using_html_table(page)

            context.close()
            return data
    except Exception as e:
        raise RuntimeError(f"웹페이지에서 표를 읽지 못했습니다: {e}") from e


# =========================
# 판단/알림 로직
# =========================

def find_nodes_below_threshold(
    node_gpu_list: list[tuple[str, int]],
    threshold: int,
) -> list[tuple[str, int]]:
    """할당 GPU 수가 기준값 이하인 노드만 골라냅니다.

    Args:
        node_gpu_list (list[tuple[str, int]]): 길이 N의 목록. (노드, 할당GPU수)
        threshold (int): 기준값. 이 값 이하이면 '조건 만족'입니다.

    Returns:
        list[tuple[str, int]]: 길이 M의 목록. (노드, 할당GPU수)
    """
    # node_gpu_list shape: (N, 2)
    matched: list[tuple[str, int]] = []
    # matched shape: (M, 2)
    for node, alloc in node_gpu_list:
        if alloc <= threshold:
            matched.append((node, alloc))
    return matched


def build_email_message(
    grafana_url: str,
    threshold: int,
    matched_nodes: list[tuple[str, int]],
) -> tuple[str, str]:
    """이메일 제목과 본문을 만듭니다.

    Args:
        grafana_url (str): 확인한 링크.
        threshold (int): 기준값.
        matched_nodes (list[tuple[str, int]]): 길이 M의 목록. (노드, 할당GPU수)

    Returns:
        tuple[str, str]: (제목, 본문)
    """
    subject = f"[GPU 알림] 할당 GPU {threshold} 이하 노드 발견 ({len(matched_nodes)}대)"

    lines: list[str] = []
    # lines shape: (L,)
    lines.append("조건을 만족하는 노드가 발견되었습니다.")
    lines.append(f"- 기준: 할당된 총 GPU 수 <= {threshold}")
    lines.append(f"- 발견 수: {len(matched_nodes)}")
    lines.append("")
    lines.append("노드 목록:")
    for node, alloc in matched_nodes:
        lines.append(f"- {node} : {alloc}")
    lines.append("")
    lines.append("확인 링크:")
    lines.append(grafana_url)

    body = "\n".join(lines)
    return subject, body


def run_check_and_notify_once(
    config: AppConfig,
    smtp_password: str,
    state: AlertState,
    headless_override: Optional[bool] = None,
) -> None:
    """1번 확인하고, 필요하면 이메일을 보낸 뒤 상태를 갱신합니다.

    동작:
    - 표에서 노드별 할당 GPU 수를 읽습니다.
    - 기준값 이하 노드가 있으면 '알림 상태'로 봅니다.
    - 직전에는 알림 상태가 아니었는데, 이번에 처음 알림 상태가 되면 이메일을 1번 보냅니다.
    - 알림 상태가 풀리면(조건 만족 노드가 없어지면) 다음 알림을 위해 상태를 초기화합니다.

    Args:
        config (AppConfig): 전체 설정값.
        smtp_password (str): SMTP 비밀번호.
        state (AlertState): 직전 상태.
        headless_override (Optional[bool]): 이번 실행에서만 headless 값을 덮어쓸 때 사용.

    Raises:
        RuntimeError: 웹페이지 읽기/이메일 전송 실패 시.
    """
    headless = config.watch.headless if headless_override is None else headless_override

    node_gpu_list = fetch_node_allocated_gpu_counts(
        url=config.watch.grafana_url,
        browser_profile_dir=Path(config.watch.browser_profile_dir),
        headless=headless,
        table_wait_seconds=config.watch.table_wait_seconds,
    )

    matched = find_nodes_below_threshold(
        node_gpu_list=node_gpu_list,
        threshold=config.watch.allocated_gpu_threshold,
    )

    alert_now = len(matched) > 0

    if alert_now and not state.alert_active:
        subject, body = build_email_message(
            grafana_url=config.watch.grafana_url,
            threshold=config.watch.allocated_gpu_threshold,
            matched_nodes=matched,
        )
        _send_email(config.email, smtp_password, subject, body)
        LOGGER.info("이메일 발송 완료. 조건 만족 노드 수=%d", len(matched))
        state.alert_active = True
        return

    if not alert_now and state.alert_active:
        LOGGER.info("조건이 해제되었습니다(다음에 다시 조건이 생기면 이메일을 보냅니다).")
        state.alert_active = False
        return

    LOGGER.info("변화 없음. 조건 만족 노드 수=%d (알림 상태=%s)", len(matched), state.alert_active)


# =========================
# 메인 루프
# =========================

def _setup_logging() -> None:
    """로그 출력을 보기 좋게 설정합니다."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def main() -> None:
    """프로그램 시작점입니다."""
    _setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true", help="최초 설정을 강제로 다시 받습니다.")
    parser.add_argument("--once", action="store_true", help="한 번만 확인하고 종료합니다.")
    parser.add_argument("--headful", action="store_true", help="브라우저 화면을 띄워서 실행합니다(로그인 저장용).")
    args = parser.parse_args()

    app_dir = _get_app_dir()
    _ensure_app_dir_exists(app_dir)

    config_path = _get_config_path()
    state_path = _get_state_path()

    if args.init and config_path.exists():
        config_path.unlink()

    config = _load_or_create_config(config_path)
    smtp_password = _get_or_ask_email_password(config)

    state = _load_state(state_path)

    # headful 옵션이 있으면 이번 실행에서만 headless를 False로 둡니다.
    headless_override = False if args.headful else None

    if args.once:
        run_check_and_notify_once(config, smtp_password, state, headless_override=headless_override)
        _save_state(state_path, state)
        return

    interval_seconds = max(1, int(config.watch.check_interval_minutes * 60))
    LOGGER.info("감시 시작: %d분마다 확인합니다.", config.watch.check_interval_minutes)

    while True:
        try:
            run_check_and_notify_once(config, smtp_password, state, headless_override=headless_override)
            _save_state(state_path, state)
        except Exception as e:
            LOGGER.error("실행 중 오류: %s", e)

        time.sleep(interval_seconds)


if __name__ == "__main__":
    main()
