import fs from 'fs';
import path from 'path';
import { validateBlueprintObject } from '../src/schemaValidator.js';

console.log('== Video Blueprint Studio Contract Test Runner ==');

const examplePath = path.join(process.cwd(), 'contracts', 'example_blueprint.json');
const schemaPath = path.join(process.cwd(), 'contracts', 'video_blueprint.schema.json');

if (!fs.existsSync(examplePath)) {
  console.error(`FAIL: Example file missing at ${examplePath}`);
  process.exit(1);
}

if (!fs.existsSync(schemaPath)) {
  console.error(`FAIL: Schema file missing at ${schemaPath}`);
  process.exit(1);
}

const exampleJson = JSON.parse(fs.readFileSync(examplePath, 'utf-8'));
const result = validateBlueprintObject(exampleJson);

if (result.valid) {
  console.log('PASS: contracts/example_blueprint.json validates against Draft 2020-12 schema and engineering invariants.');
  console.log(`- Invariant checks passed: ${result.summary.passed_rules}/${result.summary.passed_rules + result.summary.failed_rules}`);
  process.exit(0);
} else {
  console.error('FAIL: contracts/example_blueprint.json failed validation!');
  console.error('Errors:', JSON.stringify(result.errors, null, 2));
  process.exit(1);
}
