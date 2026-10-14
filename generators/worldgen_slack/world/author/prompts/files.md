/task/input.json holds this phase's inputs; when it has feedback, that is what code or a reviewer
rejected in your last document. /task/world.sqlite is a read-only SQLite copy of the workspace built so far, and
/task/schema.sql is its schema. /task/schemas.json is the JSON schema of the document to write. Write the document
to /task/<phase>.json, where <phase> is input.json's phase.

