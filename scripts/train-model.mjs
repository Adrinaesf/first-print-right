#!/usr/bin/env node
import { readFile, writeFile, mkdir, rename } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { trainModel } from '../public/ml/model.js';

const [input, output, ...extra] = process.argv.slice(2);
if (!input || !output || extra.length) {
  console.error('Usage: node scripts/train-model.mjs <dataset.json> <model.json>');
  process.exitCode = 1;
} else {
  try {
    if (resolve(input) === resolve(output)) throw new Error('Input dataset and output model paths must differ.');
    const dataset = JSON.parse(await readFile(input, 'utf8'));
    const model = trainModel(dataset);
    await mkdir(dirname(output), { recursive: true });
    const temporary = `${output}.tmp-${process.pid}`;
    await writeFile(temporary, JSON.stringify(model, null, 2) + '\n');
    await rename(temporary, output);
    console.log(`Trained on ${model.training.rows} rows from ${model.training.groups} models; held out ${model.training.holdoutRows} rows from ${model.training.holdoutModelIds.length} models.`);
    console.table(model.metrics);
    console.log(`ML scoring ${model.usable ? 'enabled' : 'disabled'}: ${model.qualityReason}`);
    console.log(`Saved ${output}`);
  } catch (error) {
    console.error(`Training failed: ${error.message}`);
    process.exitCode = 1;
  }
}
