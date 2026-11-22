-- Give flashover, foreign_object and broken_strand their own DefectCategories instead of BROKEN_INSULATOR.
-- Data only; no backend code change. Severity weight and emergency flag stay as before (0.9, emergency),
-- so severity scoring is unchanged; only the category the Analyst sees changes.
-- Idempotent: safe to run more than once. Applies to new detections only; existing rows keep their category.
BEGIN;

INSERT INTO "DefectCategories" ("Id", "CategoryCode", "CategoryName", "SeverityWeight", "IsEmergencyClass", "Description")
SELECT (SELECT max("Id") FROM "DefectCategories") + row_number() OVER (ORDER BY v.code), v.code, v.name, 0.9, true, v.description
FROM (VALUES
    ('FLASHOVER', 'Sứ phóng điện', 'Vết phóng điện trên chuỗi cách điện do AI phát hiện (nhãn insulator_flashover)'),
    ('FOREIGN_OBJECT', 'Vật lạ trên đường dây', 'Vật lạ vướng trên dây dẫn hoặc cột do AI phát hiện (nhãn foreign_object)'),
    ('BROKEN_STRAND', 'Đứt sợi dây dẫn', 'Dây dẫn bị đứt hoặc bung sợi do AI phát hiện (nhãn broken_strand)')
) AS v(code, name, description)
WHERE NOT EXISTS (SELECT 1 FROM "DefectCategories" c WHERE c."CategoryCode" = v.code);

SELECT setval(pg_get_serial_sequence('"DefectCategories"', 'Id'), (SELECT max("Id") FROM "DefectCategories"));

UPDATE "DefectCategories"
SET "Description" = 'Bát cách điện vỡ hoặc nứt do AI phát hiện (nhãn insulator_broken)'
WHERE "CategoryCode" = 'BROKEN_INSULATOR';

UPDATE "SystemSettings" s
SET "Value" = v.code, "UpdatedAt" = now()
FROM (VALUES
    ('AiLabel:insulator_flashover', 'FLASHOVER'),
    ('AiLabel:foreign_object', 'FOREIGN_OBJECT'),
    ('AiLabel:broken_strand', 'BROKEN_STRAND')
) AS v(key, code)
WHERE s."Key" = v.key AND NOT s."IsDeleted";

-- Every AiLabel mapping must resolve (same rule as scripts/ops/check_production.sh); abort otherwise.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM "SystemSettings" s
               WHERE s."Key" LIKE 'AiLabel:%' AND NOT s."IsDeleted" AND upper(coalesce(s."Value", '')) <> 'NONE'
                 AND NOT EXISTS (SELECT 1 FROM "DefectCategories" c WHERE c."CategoryCode" = s."Value")) THEN
        RAISE EXCEPTION 'AiLabel mapping without a DefectCategory';
    END IF;
END $$;

COMMIT;

SELECT s."Key", s."Value", c."CategoryName", c."SeverityWeight", c."IsEmergencyClass"
FROM "SystemSettings" s LEFT JOIN "DefectCategories" c ON c."CategoryCode" = s."Value"
WHERE s."Key" LIKE 'AiLabel:%' AND NOT s."IsDeleted" ORDER BY 1;
