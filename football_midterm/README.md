# 축구 선수 추적 중간 구현

YOLO11s의 COCO 사전학습 모델로 사람을 탐지하고 ByteTrack·BoT-SORT를 비교한 개인 프로젝트의 중간 구현이다. 실제 선수/심판 분류와 경기장 좌표 보정은 아직 구현하지 않았다.

## 포함 파일

- `mid_report.md`: 교수님 중간보고서 양식에 맞춘 보고서
- `analyze.py`: 영상 처리, 추적, CSV 저장, 분석 그래프 생성
- `render_evidence.py`: 저장한 CSV로 작은 ID 표시와 H.264 결과 영상을 만드는 선택 작업(FFmpeg 필요)
- `query_track.py`: 추적 ID와 시간 구간을 선택한 CSV·그래프 생성
- `results/`: 두 추적기의 좌표 CSV, 통계, 그래프, 결과 영상
- `requirements.txt`: 실제 실험 환경의 주요 패키지 버전
- `run.log`: 실제 분석 실행 기록
- `results/experiment.json`: 실험 조건, 데이터 해시, 라이브러리 버전, 측정 결과

## 재실행

Python 3.12 환경에서 아래와 같이 실행한다. Windows CPU 환경에서는 먼저 CPU용 PyTorch를 설치하면 된다.

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python analyze.py --video data/sample.mp4 --weights yolo11s.pt
```

원본 영상과 가중치는 ZIP에 포함하지 않는다. 참고 저장소의 Sample input video를 내려받아 `data/sample.mp4`로 저장한다. `yolo11s.pt`가 없으면 Ultralytics가 실행 시 공식 가중치 다운로드를 시도한다.

- 참고 저장소: https://github.com/abdullahtarek/football_analysis
- 샘플 영상: https://drive.google.com/file/d/1t6agoqggZKx6thamUuPAIdN_1zR9v9S_/view
- 공식 모델: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt

기본 실행은 전체 영상을 프레임 생략 없이 CPU에서 처리한다. 빠른 동작 확인은 `--limit 50`을 추가한다. 재실행은 같은 출력 폴더를 덮어쓰므로 별도 결과를 보존하려면 `--output results_new`를 지정한다.

## 선수 후보 조회

`results/bytetrack/track_summary.csv`에서 원하는 ID를 확인하고 실행한다. ID는 실제 등번호가 아니다.

```bash
python query_track.py --id 1 --start 0 --end 10 --output results/selected
```

지정한 ID가 해당 시간에 없으면 안내 후 종료한다. 범위는 시작 시각 이상, 종료 시각 미만이다. 긴 누락 구간은 이동 경로 선으로 연결하지 않는다.

## 결과 해석

- `positions.csv`: 추적기가 반환한 사람 박스의 관측값. 미탐지된 사람과 숨겨진 추적 상태는 포함하지 않는다.
- `internal_gap_frames`: 같은 ID의 최초·마지막 관측 사이에서 반환되지 않은 프레임 수. 화면 밖 이동과 가림도 포함될 수 있어 실제 미탐지율과 다르다.
- 고유 ID 수는 선수 수도, ID 변경 횟수도 아니다.
- `processing_fps`: 워밍업 후 model.track 호출 시간 기준. `end_to_end_fps`는 영상 읽기·분석용 통계·그리기·원시 영상 쓰기까지 포함하며 사후 그래프 생성·압축 시간은 제외한다.
- 비교는 추적기별 1회 순차 실행이다. 엄밀한 속도 벤치마크나 정확도 평가가 아니다.
- 화면 좌표 경로·히트맵은 카메라 이동을 포함한다. 미터 단위 이동거리, km/h, 실제 경기장 활동 영역으로 해석하지 않는다.
- BoT-SORT의 추적용 카메라 움직임 보정은 경기장 좌표 변환과 다르다.
- 데이터 한 개만 분석했으며 학습·미세조정은 수행하지 않았다. 정답 라벨이 없으므로 mAP, IDF1, HOTA, 정확한 ID switch는 계산하지 않았다.

## 제출 전

보고서에는 제공된 교수님 가이드라인의 진행 상황·결과·수정 목표·범위 및 일정·향후 계획을 반영했다. 학번·GitHub 아이디를 입력하고 수업의 제출 조건을 최종 확인해야 한다. 원본 및 결과 영상의 외부 공개 전에는 해당 영상의 이용 조건을 확인한다. 코드 작성·실행·문서 정리에 AI 도움을 받았으며, 제출자는 코드와 결과를 직접 확인하고 설명할 수 있도록 검토한다.

## 보기 편한 결과 영상 다시 만들기

기본 분석의 `annotated.mp4` 대신 작은 ID만 표시한 영상을 만들려면, FFmpeg를 설치하고 원본 영상이 있는 상태에서 다음 명령을 실행한다. 모델 추론은 반복하지 않는다. 이번 ZIP에는 이 방식으로 만든 `tracking_preview.mp4`가 들어 있다.

```bash
python render_evidence.py
```

## 완성 및 발표 목표 일정

- 11월 16일: 중간보고서 제출
- 11월 26일: 새 기능 추가 마감
- 12월 2일: 최종 평가·보고서·코드 정리 완료
- 12월 3~4일: PPT·대본·시연 영상 제작
- 12월 5~6일: 리허설과 예비일
- 12월 7일: 최종 발표 목표일(준비 마감 기준으로 설정, 실제 공지 시 조정)
