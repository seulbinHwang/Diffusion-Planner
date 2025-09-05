import glob
import logging
import pathlib
from concurrent.futures import Future
from typing import List, Optional, Union, Tuple
from nuplan_extent.planning.simulation.observation.world_model_agents import WorldModelAgents
import cv2
from abc import ABC, abstractmethod
import torch  # noqa: F401  # (현재 코드에서는 직접 사용하지 않지만 외부 인터페이스 호환을 위해 유지)
import numpy as np  # 타입 힌트/설명용

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.callback.abstract_callback import AbstractCallback
from nuplan.planning.simulation.history.simulation_history import (
    SimulationHistory, SimulationHistorySample)
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.planner.ml_planner.ml_planner import MLPlanner

logger = logging.getLogger(__name__)


def save_video(
    frame_size_hw: Tuple[int, int],
    frames_rgb: List[np.ndarray],
    output_path: Union[str, pathlib.Path],
    database_interval: float,
) -> None:
    """이미지 프레임 시퀀스를 비디오 파일로 저장한다.

    이 함수는 시뮬레이션 중 렌더된 BEV/피처 이미지를 모아 하나의 비디오로 저장한다.
    입력 프레임은 **RGB 색상 순서**를 가정하며, OpenCV `VideoWriter`에 기록하기 전
    내부적으로 **BGR**로 변환한다. 프레임 크기가 서로 다를 경우, 첫 프레임의 크기에
    맞춰 **리사이즈**하여 일관된 해상도로 저장한다.

    Args:
        frame_size_hw (Tuple[int, int]):
            비디오 해상도 (height, width). 예: `(H, W)`.
            `VideoWriter` 초기화에는 `(W, H)` 순서가 필요하므로 내부에서 변환한다.
        frames_rgb (List[np.ndarray]):
            프레임 리스트. 각 원소는 `shape=(H, W, 3)`, `dtype=uint8`, **RGB** 색상 순서를 권장한다.
            (GRAY/4채널 입력도 허용하며, 내부에서 BGR 3채널로 변환 후 기록한다.)
        output_path (Union[str, pathlib.Path]):
            저장할 비디오 경로. 확장자는 `.webm`(권장) 또는 `.mp4` 등을 사용할 수 있다.
            `.webm`의 경우 VP9/VP8 코덱을 우선 시도한다.
        database_interval (float):
            프레임 간 시간 간격(초). FPS는 `1.0 / database_interval`로 계산한다.
            값이 0 또는 음수일 경우 기본 FPS=10.0을 사용한다.

    Returns:
        None

    Raises:
        RuntimeError: 지원되는 코덱으로 `VideoWriter`를 열지 못한 경우.
        ValueError: 유효한 프레임이 없거나, 프레임 형태가 잘못된 경우.

    Notes:
        - `.webm` 컨테이너는 일반적으로 VP8/VP9 코덱을 요구한다. 시스템 환경에 따라
          OpenCV가 해당 코덱을 지원하지 않을 수 있으며, 그 경우 `ffmpeg` 설치/설정이 필요할 수 있다.
        - 프레임 크기가 제각각일 경우, 첫 프레임의 `(H, W)`에 맞춰 모두 **리사이즈**하여 저장한다.
        - `frames_rgb`가 빈 리스트면 아무 것도 저장하지 않고 `ValueError`를 발생시킨다.
    """
    # 입력 검증
    if not frames_rgb:
        raise ValueError("save_video: 저장할 프레임이 비어 있습니다(frames_rgb=[]).")

    # FPS 계산
    fps: float = 10.0 if (database_interval is None or
                          database_interval <= 0) else float(1.0 /
                                                             database_interval)

    # 경로/확장자 정리
    path = pathlib.Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    # 목표 해상도 (VideoWriter는 (width, height) 순서)
    target_h, target_w = int(frame_size_hw[0]), int(frame_size_hw[1])
    target_size_wh = (target_w, target_h)

    # 코덱 후보 설정
    if suffix == ".webm":
        codec_candidates = ["VP90", "VP80"]  # VP9 → VP8
    elif suffix in [".mp4", ".m4v"]:
        codec_candidates = ["avc1", "H264", "mp4v"]
    else:
        # 기타 확장자는 광범위하게 시도
        codec_candidates = ["VP90", "VP80", "avc1", "H264", "mp4v", "MJPG"]

    writer = None
    chosen_codec = None
    for codec in codec_candidates:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(path),
                                 fourcc,
                                 fps,
                                 target_size_wh,
                                 isColor=True)
        if writer is not None and writer.isOpened():
            chosen_codec = codec
            break
        if writer is not None:
            writer.release()
            writer = None

    if writer is None or not writer.isOpened():
        raise RuntimeError(
            f"save_video: VideoWriter 초기화 실패. 경로={path}, FPS={fps}, size={target_size_wh}, "
            f"시도한 코덱={codec_candidates}. 시스템의 ffmpeg/코덱 지원 상태를 확인하세요.")

    if chosen_codec is not None:
        logger.info(
            f"save_video: writer opened. path={path}, fps={fps:.3f}, size={target_size_wh}, codec={chosen_codec}"
        )

    # 프레임 기록
    frame_count = 0
    for idx, img in enumerate(frames_rgb):
        if img is None:
            logger.warning(f"save_video: idx={idx} 프레임이 None 입니다. 스킵합니다.")
            continue

        # 채널/형태 정규화
        if img.ndim == 2:
            # GRAY → BGR
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:
            # RGBA → RGB → BGR
            img_rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        elif img.ndim == 3 and img.shape[2] == 3:
            # RGB → BGR
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError(
                f"save_video: idx={idx} 예상치 못한 프레임 shape={getattr(img, 'shape', None)}. "
                "지원되는 형태: (H,W), (H,W,3), (H,W,4)")

        # 사이즈 정규화
        if (img_bgr.shape[1], img_bgr.shape[0]) != target_size_wh:
            img_bgr = cv2.resize(img_bgr,
                                 target_size_wh,
                                 interpolation=cv2.INTER_AREA)

        writer.write(img_bgr)
        frame_count += 1

    writer.release()

    if frame_count == 0:
        raise ValueError("save_video: 기록된 유효 프레임이 없습니다(모든 프레임이 None/형태 오류).")
    logger.info(
        f"save_video: 저장 완료. path={path}, frames={frame_count}, fps={fps:.3f}")


class SimulationFeatureVideoCallback(AbstractCallback):
    """시뮬레이션 중 선택한 시나리오의 **피처 렌더링 이미지**를 저장하고,
    시뮬레이션 종료 시 해당 이미지들을 하나의 **동영상**으로 합치는 콜백.

    이 콜백은 다음과 같은 작업 흐름을 갖는다.

    - 초기화 단계에서 각 시나리오별 결과 저장 폴더를 생성한다.
    - 각 스텝 시작 시(`on_step_start`) 현재 스텝 인덱스를 기반으로
      렌더링 이미지(예: BEV 피처, 네트워크 입력 등)가 저장될 파일 경로를
      `observations.set_vis_features(...)`로 전달한다.
    - 시뮬레이션 종료 시(`on_simulation_end`) 해당 폴더의 PNG들을 시간 순서대로 로드하고
      `save_video(...)`를 호출하여 `.webm` 동영상으로 내보낸다.

    Note:
        - 어떤 시나리오를 시각화할지는 `visualized_scenario_tokens` 또는
          `visualize_all_scenarios` 플래그로 제어한다.
        - 이미지 파일 접미사는 `image_subfix`로 지정하며(예: `.png`),
          실제 파일명은 `{iteration_index:04d}{subfix}` 형태로 저장된다.
    """

    def __init__(
        self,
        simulation_directory: Union[str, pathlib.Path],
        videos_output_dir: Union[str, pathlib.Path],
        feature_log_dir: Union[str, pathlib.Path],
        visualized_scenario_tokens: Optional[List[str]] = [],
        visualize_all_scenarios: bool = False,
        bev_range: List[float] = [-56., -56., 56., 56.],
        image_subfix: str = ".png",
    ):
        """콜백 인스턴스를 생성한다.

        Args:
            simulation_directory (Union[str, pathlib.Path]):
                시뮬레이션 산출물이 기록될 **루트 디렉터리**.
            videos_output_dir (Union[str, pathlib.Path]):
                동영상 파일들을 저장할 **상대 경로** 또는 **절대 경로**.
                실제 저장 경로는 `simulation_directory / videos_output_dir`.
            feature_log_dir (Union[str, pathlib.Path]):
                렌더링된 피처 이미지들을 저장할 **상대/절대 경로**.
                실제 저장 경로는 `simulation_directory / feature_log_dir`.
            visualized_scenario_tokens (Optional[List[str]]):
                시각화할 시나리오 토큰 목록. 지정되지 않았거나 빈 리스트일 경우,
                `visualize_all_scenarios=True`인 경우에 한해 모든 시나리오를 시각화한다.
            visualize_all_scenarios (bool):
                `True`일 경우 **모든 시나리오**를 시각화 대상으로 처리한다.
            image_subfix (str):
                저장될 이미지 파일의 접미사(확장자). 예: `.png`, `.jpg`.

        Notes:
            - `visualized_scenario_tokens`에 기본값으로 **빈 리스트**를 사용하는 것은
              파이썬에서 권장되지 않는 패턴이지만(가변 기본 인자), 이 콜백은 외부 설정(YAML 등)
              로드 시 일반적으로 명시적으로 값이 주입되므로 그대로 유지한다.
        """
        assert isinstance(visualized_scenario_tokens,
                          list), "visualized_scenario_tokens must be a list"

        self._futures: List[Future[None]] = []
        self._visualized_scenario_tokens = visualized_scenario_tokens
        self._visualize_all_scenarios = visualize_all_scenarios
        self._videos_output_path = pathlib.Path(
            simulation_directory) / videos_output_dir
        self._feature_log_directory = pathlib.Path(
            simulation_directory) / feature_log_dir
        self._subfix = image_subfix

    def on_initialization_start(self, setup: SimulationSetup,
                                planner: AbstractPlanner) -> None:
        """시뮬레이션 초기화 시작 시 호출된다.

        필요한 경우, **시나리오별 결과 저장 디렉터리**를 생성한다.
        (예: `{feature_log_dir}/{planner}/{scenario_type}/{log_name}/{scenario_name}/features`)

        Args:
            setup (SimulationSetup): 시뮬레이션 설정/컨텍스트.
            planner (AbstractPlanner): 초기화 전 상태의 플래너 인스턴스.

        Returns:
            None
        """
        scenario_token = setup.scenario.token
        if self._visualize_all_scenarios or (
                scenario_token in self._visualized_scenario_tokens):
            scenario_directory = self._get_scenario_folder(
                planner.name(), setup.scenario)
            feature_log_directory = scenario_directory / "features"
            feature_log_directory.mkdir(exist_ok=True, parents=True)

    def on_initialization_end(self, setup: SimulationSetup,
                              planner: AbstractPlanner) -> None:
        """시뮬레이션 초기화 종료 시 호출된다.

        현재 구현에서는 별도의 동작을 수행하지 않는다. 후속 단계(시뮬레이션 시작)에서
        시각화용 디렉터리/파일 경로가 사용된다.

        Args:
            setup (SimulationSetup): 시뮬레이션 설정/컨텍스트.
            planner (AbstractPlanner): 초기화가 끝난 플래너.

        Returns:
            None
        """

    def on_simulation_start(self, setup: SimulationSetup) -> None:
        """시뮬레이션 시작 시점에 호출된다.

        현재 구현에서는 별도의 동작을 수행하지 않는다. 스텝 시작 시점에
        각 스텝의 이미지 저장 경로를 지정한다.

        Args:
            setup (SimulationSetup): 시뮬레이션 설정/컨텍스트.

        Returns:
            None
        """

    def on_step_start(self, setup: SimulationSetup,
                      planner: AbstractPlanner) -> None:
        """각 시뮬레이션 스텝 시작 시 호출된다.

        시각화 대상 시나리오에 한하여, **현재 스텝 인덱스**를 기반으로
        렌더링 이미지가 저장될 파일 경로를 `observations.set_vis_features(...)`에 전달한다.
        현재는 `MLPlanner`에 대해서만 경로 전달이 구현되어 있으며,
        다른 플래너 타입에 대해서는 `NotImplementedError`를 발생시킨다.

        Args:
            setup (SimulationSetup): 시뮬레이션 설정/컨텍스트.
            planner (AbstractPlanner): 현재 사용 중인 플래너.

        Raises:
            NotImplementedError: 플래너가 `MLPlanner`가 아닌 경우.

        Returns:
            None
        """
        scenario_token = setup.scenario.token
        if self._visualize_all_scenarios or (
                scenario_token in self._visualized_scenario_tokens):
            scenario_directory = self._get_scenario_folder(
                planner.name(), setup.scenario)
            feature_log_directory = scenario_directory / "features"
            simulation_iteration_index = setup.time_controller.get_iteration(
            ).index  # int

            # 각 스텝의 렌더링 저장 경로를 Planner/Observation에 전달
            if isinstance(setup.observations, WorldModelAgents):
                setup.observations.set_vis_features(
                    is_vis_features=True,
                    vis_features_path=feature_log_directory /
                    "{:04d}{}".format(simulation_iteration_index, self._subfix),
                )
            else:
                raise NotImplementedError(
                    "현재 SimulationFeatureVideoCallback은 MLPlanner만 지원합니다.")
        else:
            setup.observations.set_vis_features(is_vis_features=False,
                                                vis_features_path=None)

    def on_planner_start(self, setup: SimulationSetup,
                         planner: AbstractPlanner) -> None:
        """플래너가 현재 스텝의 계획(trajectory 계산)을 시작하기 직전에 호출된다.

        현재 구현에서는 별도의 동작을 수행하지 않는다.

        Args:
            setup (SimulationSetup): 시뮬레이션 설정/컨텍스트.
            planner (AbstractPlanner): 플래너 인스턴스.

        Returns:
            None
        """

    def on_planner_end(self, setup: SimulationSetup, planner: AbstractPlanner,
                       trajectory: AbstractTrajectory) -> None:
        """플래너가 현재 스텝의 계획을 끝내고 궤적을 반환한 직후 호출된다.

        현재 구현에서는 별도의 동작을 수행하지 않는다. 필요 시 플래너 출력(trajectory)을
        사용해 추가 시각화 메타데이터를 기록하도록 확장할 수 있다.

        Args:
            setup (SimulationSetup): 시뮬레이션 설정/컨텍스트.
            planner (AbstractPlanner): 플래너 인스턴스.
            trajectory (AbstractTrajectory): 이번 스텝에서 계산된 궤적.

        Returns:
            None
        """

    def on_step_end(self, setup: SimulationSetup, planner: AbstractPlanner,
                    sample: SimulationHistorySample) -> None:
        """각 스텝이 종료될 때 호출된다.

        현재 구현에서는 별도의 동작을 수행하지 않는다. 필요 시 `sample`에 포함된
        상태/관측/액션을 참조하여 부가 로그를 남기도록 확장할 수 있다.

        Args:
            setup (SimulationSetup): 시뮬레이션 설정/컨텍스트.
            planner (AbstractPlanner): 플래너 인스턴스.
            sample (SimulationHistorySample): 이번 스텝의 히스토리 샘플.

        Returns:
            None
        """

    def on_simulation_end(self, setup: SimulationSetup,
                          planner: AbstractPlanner,
                          history: SimulationHistory) -> None:
        """시뮬레이션 종료 시 호출된다.

        시각화 대상 시나리오의 **피처 이미지**들을 시간 순서대로 로드한 뒤,
        `save_video(...)`를 호출하여 `.webm` 비디오를 생성/저장한다.
        이미지 로드는 OpenCV로 수행하며(BGR), 로드 직후 RGB로 변환하여 `frames_rgb` 리스트에 넣는다.
        (`save_video` 내부에서 다시 BGR로 변환 후 기록한다.)

        Args:
            setup (SimulationSetup): 시뮬레이션 설정/컨텍스트.
            planner (AbstractPlanner): 플래너 인스턴스.
            history (SimulationHistory): 전체 시뮬레이션 히스토리(미사용).

        Returns:
            None
        """
        scenario_token = setup.scenario.token
        database_interval = setup.scenario.database_interval

        if self._visualize_all_scenarios or (
                scenario_token in self._visualized_scenario_tokens):
            video_images: List[np.ndarray] = []
            scenario_directory = self._get_scenario_folder(
                planner.name(), setup.scenario)
            feature_log_directory = scenario_directory / "features"

            # 스텝 인덱스 순서대로 정렬된 이미지 경로 로드
            feature_paths = sorted(glob.glob(str(feature_log_directory / "*")))
            for p in feature_paths:
                image_bgr = cv2.imread(p)
                if image_bgr is None:
                    logger.warning(f"on_simulation_end: 이미지 로드 실패. path={p}")
                    continue
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                video_images.append(image_rgb)

            if not video_images:
                logger.warning(
                    "on_simulation_end: 비디오로 합칠 유효한 이미지가 없습니다. 스킵합니다.")
                return

            video_save_path = pathlib.Path(self._videos_output_path /
                                           "feature_selected_scenarios")
            video_name = scenario_token + ".webm"
            video_save_path.mkdir(parents=True, exist_ok=True)

            # (H, W), frames, output_path, fps(source: database_interval)
            save_video(
                video_images[0].shape[:2],
                video_images,
                video_save_path / video_name,
                database_interval,
            )

    def _get_scenario_folder(self, planner_name: str,
                             scenario: AbstractScenario) -> pathlib.Path:
        """시나리오별 산출물(이미지/비디오)을 저장할 디렉터리 경로를 생성한다.

        디렉터리 구조는 다음과 같다.

        ```
        {feature_log_dir}/{planner_name}/{scenario_type}/{log_name}/{scenario_name}/
        ```

        Args:
            planner_name (str): 플래너 이름. 예: `"MLPlanner"`.
            scenario (AbstractScenario): 시나리오 객체. `scenario_type`, `log_name`, `scenario_name`을 사용한다.

        Returns:
            pathlib.Path: 위 규칙에 따라 구성된 디렉터리 경로.
        """
        return (self._feature_log_directory / planner_name /
                scenario.scenario_type / scenario.log_name /
                scenario.scenario_name  # type: ignore
               )
