# BE integration — inspected 2026-10-28

## Recommended network layout

BE -> RabbitMQ (private) -> bridge worker (same Docker network/VPS) -> HTTPS Modal GPU -> bridge -> RabbitMQ result -> AIInspectionService -> database/frontend.

Different public IPs do NOT require exposing RabbitMQ. Worker needs outbound HTTPS to Modal plus private connectivity to broker and media storage. Modal never connects back to broker. GPU containers scale to zero; a long-lived broker consumer belongs beside BE, not inside the GPU endpoint.

## Implementation

Bridge đã có ở `../bridge/`: RabbitMQ durable inbox, Modal HTTP client, image/video sampling, result outbox và compose. Xem README bridge để cấu hình và giới hạn. Chưa chạy end-to-end trên môi trường thật. Device-key/idempotency BE contract ở `../BE_HANDOFF.md`.

## Existing BE contract

Source repo: UavPms_Org/Services/AIInspectionService.

Request topology: Infrastructure/Messaging/AIAnalysisRequestTopology.cs.
Exchange identity-exchange (topic/durable).
Consume only ai.analysis.server.image.requested and ai.analysis.server.video.requested for this cloud service. Routing keys identity.event.aianalysisrequestedevent.server.image and .server.video. Queue arguments must match BE x-dead-letter-exchange=ai.analysis.dlx; do not redeclare with incompatible arguments. BE routes YOLO11 or EDGE to edge queues, everything else to server queues.

Request JSON from Domain/Contracts/AIAnalysisRequestedEvent.cs: RequestId, FileUrl, MediaType, AnalysisType, Notes, UploadedBy, RequestedAt, MediaId, MissionId, AssetId, PreferredModel. Publisher uses System.Text.Json default PascalCase.

Result: exchange identity-exchange, key identity.event.aianalysisresultevent, queue ai.analysis.result. Domain/Contracts/AIAnalysisResultEvent.cs defines EventId, CorrelationId, AnalysisId, InspectionId, MediaId, MissionId, AssetId, Status, ModelName, ModelVersion, ProcessingTimeMs, Results, VideoMetadata, RawResult, ErrorCode, ErrorMessage, ProcessedAt.

Critical mappings:
- AnalysisId = request.RequestId (consumer maps this to command RequestId).
- Preserve MediaId/MissionId; current result event has no AssetId. TowerId is mapped per detection from TowerId or legacy AssetId.
- Status Processing/Completed/Failed; Completed needs model name and detection IDs.
- Results[].BoundingBox is normalized top-left X/Y/Width/Height, NOT pixel xyxy.
- CategoryCode must exist in BE DefectCategory; Class can carry original YOLO label. Do not assume six model labels equal database category codes. Healthy insulator should not automatically create a defect.
- Video frame detections require FrameIndex and Timestamp seconds/TimestampMs. Sampling is not tracking, and skipped frames can contain missed defects.

## Reliability requirements for the bridge

Persist job keyed by RequestId before work. Persist results and EventId before publish; reuse on redelivery rather than generate new events/detections. Publish with confirms/mandatory, ACK request only after a durable result is confirmed or durable handoff is complete. DB outbox handles crash windows; broker provides at-least-once, not automatic exactly-once. Bounded transient retries; final failure event then dead-letter policy. Log request ID, media ID, stage, attempt, model hash, frame counts and timing; never credentials or signed URL query strings.

Videos: recommended initial sampling1 frame/sec, configurable max duration/frames/bytes, clearly expose sampling metadata. Long videos must not hold a single synchronous HTTP request: individual bounded frame calls initially, or durable submit/poll jobs on Modal later. Keep broker heartbeat alive and account for delivery-ack timeout; use durable job handoff for long jobs. Avoid arbitrary FileUrl fetching: restrict schemes/hosts, validate redirects, limit bytes/time, use credentials from environment only.

## If RabbitMQ is unavailable entirely

Use BE database-backed jobs/outbox with a polling worker and lease/heartbeat (Pending/Processing/Completed/Failed, Attempts, NextAttemptAt, LeaseUntil). Frontend API remains asynchronous. An in-memory queue or FastAPI BackgroundTasks alone does not preserve work across restarts. Modal submit/spawn + poll can handle GPU work, but BE must persist RequestId/call_id and completion delivery. This alternative requires BE work; not enabled by switching an environment flag in current code.

## Needed to make it live

1. Worker host/network: same Docker network as BE is recommended; connection address/vhost and credentials configured locally (do not paste secrets into chat).
2. One representative FileUrl shape and media-access policy: local private URL, storage URL or signed URL, expiry, worker access.
3. Actual DefectCategory mapping for the five defect classes, plus healthy-insulator handling.
4. Video requirements: max length/size and sampling rate; default proposal1fps pending user preference.
5. Downloaded curated_yolo26m checkpoint, authenticated Modal profile bunpmc/main, and endpoint/proxy tokens after deploy.

Official Modal timeout/job polling guidance: https://modal.com/docs/guide/webhook-timeouts
