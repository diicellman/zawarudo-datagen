-- Persona seed sample (Nemotron-Personas-USA, CC-BY-4.0) and real per-user typing statistics
-- (spencer/software_slacks) for the [personas] config. Deterministic: sources are revision-pinned and every
-- selection orders by md5. Run from the repository root: duckdb < scripts/worldgen_slack/personas.sql

SET VARIABLE sample_size = 25000;
SET VARIABLE occupation_floor = 50;

CREATE MACRO lead_name(s) AS trim(regexp_extract(s, '^((?:[A-Z][A-Za-z''’.\-]*[ ]?)+)', 1));
-- The *_list columns hold Python list literals; an item is quoted with ' unless it contains one.
CREATE MACRO py_list(s) AS list_transform(regexp_extract_all(s, '''[^'']*''|"[^"]*"'), x -> x[2:-2]);

CREATE TABLE office AS
SELECT *
FROM 'hf://datasets/nvidia/Nemotron-Personas-USA@5b4cd35ab46490c1da1bd2b5a2324d6f871be180/data/*.parquet'
WHERE age BETWEEN 22 AND 67 AND occupation IN (
    'accountant_or_auditor', 'actuary', 'administrative_services_manager', 'advertising_or_promotions_manager',
    'advertising_sales_agent', 'architect', 'architectural_or_civil_drafter', 'architectural_or_engineering_manager',
    'billing_or_posting_clerk', 'bookkeeping_accounting_or_auditing_clerk', 'budget_analyst', 'chief_executive',
    'civil_engineer', 'claims_adjuster_appraiser_examiner_or_investigator',
    'compensation_benefits_or_job_analysis_specialist', 'compensation_or_benefits_manager', 'compliance_officer',
    'computer_hardware_engineer', 'computer_network_architect', 'computer_occupation',
    'computer_or_information_research_scientist', 'computer_or_information_systems_manager', 'computer_programmer',
    'computer_support_specialist', 'computer_systems_analyst', 'credit_analyst', 'customer_service_representative',
    'database_administrator_or_architect', 'designer', 'editor',
    'electrical_or_electronic_engineering_technologist_or_technician', 'electrical_or_electronics_engineer',
    'engineer', 'engineering_technologist_or_technician', 'executive_secretary_or_executive_administrative_assistant',
    'facilities_manager', 'financial_manager', 'first_line_supervisor_of_office_or_administrative_support_worker',
    'general_or_operations_manager', 'graphic_designer', 'human_resources_assistant', 'human_resources_manager',
    'human_resources_worker', 'industrial_engineer_including_health_or_safety', 'information_security_analyst',
    'insurance_underwriter', 'lawyer', 'logistician', 'management_analyst', 'manager',
    'market_research_analyst_or_marketing_specialist', 'marketing_manager', 'mathematical_science_occupation',
    'mathematician', 'mechanical_engineer', 'network_or_computer_systems_administrator', 'office_clerk_general',
    'operations_research_analyst', 'paralegal_or_legal_assistant', 'payroll_or_timekeeping_clerk',
    'project_management_specialist', 'public_relations_or_fundraising_manager', 'public_relations_specialist',
    'purchasing_agent', 'purchasing_manager', 'receptionist_or_information_clerk', 'sales_engineer', 'sales_manager',
    'sales_representative_of_services', 'sales_representative_wholesale_or_manufacturing',
    'secretary_or_administrative_assistant', 'software_developer', 'software_quality_assurance_analyst_or_tester',
    'statistician', 'technical_writer', 'training_or_development_manager', 'training_or_development_specialist',
    'web_developer', 'web_or_digital_interface_designer'
);

-- The dataset has no name column: every persona text opens with the name. Keep rows where at least 3 of 5 texts
-- agree on a two- or three-word name, and keep one person per name.
CREATE TABLE named AS
WITH votes AS (
    SELECT *, [lead_name(persona), lead_name(professional_persona), lead_name(sports_persona),
               lead_name(arts_persona), lead_name(culinary_persona)] AS leads
    FROM office
), voted AS (
    SELECT *, list_filter(leads, x -> x LIKE '% %' AND x NOT LIKE '% % % %') AS names FROM votes
)
SELECT * EXCLUDE (leads, names), list_aggregate(names, 'mode') AS name
FROM voted
WHERE len(list_filter(names, x -> x = list_aggregate(names, 'mode'))) >= 3
QUALIFY row_number() OVER (PARTITION BY name ORDER BY md5(uuid)) = 1;

CREATE TABLE sample AS
WITH quota AS (
    SELECT occupation, count(*) AS available,
           least(count(*), greatest(getvariable('occupation_floor'),
                 round(count(*) * getvariable('sample_size') / sum(count(*)) OVER ()))) AS take
    FROM named GROUP BY occupation
)
SELECT n.* FROM named n JOIN quota q USING (occupation)
QUALIFY row_number() OVER (PARTITION BY occupation ORDER BY md5(uuid)) <= q.take;

-- Home timezone by state; a state spanning two zones takes the one most of its people live in.
CREATE TABLE zones (state VARCHAR, timezone VARCHAR);
INSERT INTO zones VALUES
    ('AK', 'America/Anchorage'), ('AL', 'America/Chicago'), ('AR', 'America/Chicago'), ('AZ', 'America/Phoenix'),
    ('CA', 'America/Los_Angeles'), ('CO', 'America/Denver'), ('CT', 'America/New_York'),
    ('DC', 'America/New_York'), ('DE', 'America/New_York'), ('FL', 'America/New_York'), ('GA', 'America/New_York'),
    ('HI', 'Pacific/Honolulu'), ('IA', 'America/Chicago'), ('ID', 'America/Boise'), ('IL', 'America/Chicago'),
    ('IN', 'America/Indiana/Indianapolis'), ('KS', 'America/Chicago'), ('KY', 'America/New_York'),
    ('LA', 'America/Chicago'), ('MA', 'America/New_York'), ('MD', 'America/New_York'), ('ME', 'America/New_York'),
    ('MI', 'America/Detroit'), ('MN', 'America/Chicago'), ('MO', 'America/Chicago'), ('MS', 'America/Chicago'),
    ('MT', 'America/Denver'), ('NC', 'America/New_York'), ('ND', 'America/Chicago'), ('NE', 'America/Chicago'),
    ('NH', 'America/New_York'), ('NJ', 'America/New_York'), ('NM', 'America/Denver'),
    ('NV', 'America/Los_Angeles'), ('NY', 'America/New_York'), ('OH', 'America/New_York'),
    ('OK', 'America/Chicago'), ('OR', 'America/Los_Angeles'), ('PA', 'America/New_York'),
    ('PR', 'America/Puerto_Rico'), ('RI', 'America/New_York'), ('SC', 'America/New_York'),
    ('SD', 'America/Chicago'), ('TN', 'America/Chicago'), ('TX', 'America/Chicago'), ('UT', 'America/Denver'),
    ('VA', 'America/New_York'), ('VT', 'America/New_York'), ('WA', 'America/Los_Angeles'),
    ('WI', 'America/Chicago'), ('WV', 'America/New_York'), ('WY', 'America/Denver');

COPY (
    SELECT uuid, name, sex, age, marital_status, education_level, nullif(bachelors_field, '') AS bachelors_field,
           occupation, city, state, country, timezone, persona, professional_persona, cultural_background,
           py_list(skills_and_expertise_list) AS skills, py_list(hobbies_and_interests_list) AS hobbies
    FROM sample JOIN zones USING (state) ORDER BY uuid
) TO 'data/seeds/personas-usa.jsonl' (FORMAT json);

-- The corpus stores every message three times; each statistic counts a message once.
CREATE TABLE slack AS
SELECT DISTINCT workspace, channel, ts, "user", text
FROM 'hf://datasets/spencer/software_slacks@2e889c0d2b754d96a396fbc1e4b16a37f19e7d7a/data/*.parquet';

-- How real people type: surface statistics of every user with at least 30 non-empty messages. Users are keyed
-- by a digest, so no source identifier is copied.
COPY (
    WITH messages AS (
        SELECT workspace, "user", trim(text) AS text, len(string_split(trim(text), ' ')) AS words
        FROM slack
        WHERE "user" IS NOT NULL AND trim(coalesce(text, '')) <> ''
    )
    SELECT md5(workspace || '/' || "user")[:12] AS id, count(*) AS messages,
           median(words) AS median_words,
           round(avg((words <= 4)::int), 3) AS short_share,
           round(avg((words > 20)::int), 3) AS long_share,
           round(avg(contains(text, '?')::int), 3) AS question_share,
           round(avg(regexp_matches(text, '^[a-z]')::int), 3) AS lowercase_share,
           round(avg(regexp_matches(text, ':[a-z0-9_+\-]+:|\p{So}')::int), 3) AS emoji_share
    FROM messages GROUP BY workspace, "user" HAVING count(*) >= 30 ORDER BY id
) TO 'data/seeds/typing-profiles.jsonl' (FORMAT json);

-- How long people take to answer: the seconds between consecutive messages of a channel, as 1001 quantiles.
COPY (
    WITH gaps AS (
        SELECT epoch(ts::TIMESTAMP - lag(ts::TIMESTAMP) OVER (PARTITION BY workspace, channel ORDER BY ts::TIMESTAMP)) AS s
        FROM slack WHERE ts IS NOT NULL
    )
    SELECT quantile_cont(s, list_transform(range(1001), x -> x / 1000)) AS quantiles FROM gaps WHERE s IS NOT NULL
) TO 'data/seeds/reply-gaps.json' (FORMAT json);
