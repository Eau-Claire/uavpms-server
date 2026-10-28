"""BE PascalCase contracts; same RequestId and persisted EventId across delivery retries."""
import math
from datetime import datetime, timezone
from uuid import UUID, uuid4, uuid5, NAMESPACE_URL

def validate(request):
    for key in ('RequestId', 'MediaId', 'MissionId'):
        if UUID(str(request[key])).int == 0: raise ValueError('Empty identifier')
    if request['MediaType'].lower() not in ('image', 'video'): raise ValueError('Unsupported MediaType')
    if not isinstance(request['FileUrl'], str): raise ValueError('Missing FileUrl')

def event(request, results=None, error=None, elapsed=0, metadata=None):
    return dict(EventId=str(uuid4()), CorrelationId=request['RequestId'], AnalysisId=request['RequestId'],
                InspectionId='00000000-0000-0000-0000-000000000000', MediaId=request['MediaId'],
                MissionId=request['MissionId'], Status='Failed' if error else 'Completed',
                ModelName='curated_yolo26m', ModelVersion='deployment-v1', ProcessingTimeMs=round(elapsed),
                Results=results or [], VideoMetadata=metadata, ErrorCode=error,
                ErrorMessage='Worker could not complete job' if error else None,
                ProcessedAt=datetime.now(timezone.utc).isoformat())

def detections(request, output, mapping, frame_index=None, timestamp=None):
    width, height = output['width'], output['height']
    if not (width > 0 and height > 0): raise ValueError('Invalid image dimensions')
    results=[]
    for index, detection in enumerate(output['detections']):
        category=mapping[detection['class_name']]
        if category is None: continue
        x1,y1,x2,y2=detection['bbox_xyxy']
        score=detection['confidence']
        if not all(math.isfinite(v) for v in (x1,y1,x2,y2,score)) or not 0<=score<=1: raise ValueError('Invalid prediction')
        x1,x2=[min(1,max(0,x/width)) for x in (x1,x2)]
        y1,y2=[min(1,max(0,y/height)) for y in (y1,y2)]
        if x2<=x1 or y2<=y1: raise ValueError('Empty box')
        results.append(dict(Id=str(uuid5(NAMESPACE_URL,f"{request['RequestId']}:{frame_index}:{index}")),
            CategoryCode=category, Class=detection['class_name'], Confidence=score,
            BoundingBox=dict(X=x1,Y=y1,Width=x2-x1,Height=y2-y1),
            FrameIndex=frame_index, Timestamp=timestamp, TimestampMs=None if timestamp is None else round(timestamp*1000),
            TowerId=request.get('TowerId') or request.get('AssetId')))
    return results
