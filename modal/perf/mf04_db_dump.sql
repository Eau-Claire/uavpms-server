-- Read-only dump for mf04_db_checks.py: media GPS and stored AI detections of the e2e test mission.
-- Set the mission with: psql -v mission=<uuid> ...
SELECT json_build_object(
  'media', (SELECT coalesce(json_agg(json_build_object(
        'id', m."Id",
        'lat', CASE WHEN m."CaptureLocation" IS NULL THEN NULL ELSE ST_Y(m."CaptureLocation") END,
        'lon', CASE WHEN m."CaptureLocation" IS NULL THEN NULL ELSE ST_X(m."CaptureLocation") END)), '[]')
     FROM "InspectionMedia" m WHERE m."MissionId" = :'mission'),
  'anomalies', (SELECT coalesce(json_agg(json_build_object(
        'media_id', a."MediaId",
        'category', c."CategoryCode",
        'box', json_build_array((a."BoundingBox"->>'x')::float, (a."BoundingBox"->>'y')::float,
                                (a."BoundingBox"->>'width')::float, (a."BoundingBox"->>'height')::float),
        'location', CASE WHEN a."Location" IS NULL THEN NULL ELSE ST_AsText(a."Location") END)), '[]')
     FROM "DetectedAnomalies" a
     JOIN "InspectionMedia" m ON m."Id" = a."MediaId"
     JOIN "DefectCategories" c ON c."Id" = a."CategoryId"
     WHERE m."MissionId" = :'mission' AND a."SupersededById" IS NULL AND NOT a."IsDeleted")
);
