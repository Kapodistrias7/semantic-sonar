// Generates embeddings.json: MiniLM-L6-v2 vectors (mean-pooled, L2-normalized)
// for every word in the lexicon, rounded to 4 decimals.
import { pipeline, env } from '@huggingface/transformers';
import { writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { LEXICON } from './words.mjs';

env.allowRemoteModels = false;
env.localModelPath = fileURLToPath(new URL('../models/', import.meta.url));

const embedder = await pipeline('feature-extraction', 'all-MiniLM-L6-v2');

const out = {};
for (let i = 0; i < LEXICON.length; i++) {
  const word = LEXICON[i];
  try {
    const res = await embedder(word, { pooling: 'mean', normalize: true });
    out[word] = Array.from(res.data, (v) => Math.round(v * 1e4) / 1e4);
  } catch (err) {
    console.error(`\nSKIP "${word}": ${err.message}`);
  }
  if (i % 25 === 0) process.stdout.write(`\r${i}/${LEXICON.length}`);
}

const payload = JSON.stringify({ model: 'all-MiniLM-L6-v2', dims: 384, words: out });
writeFileSync(new URL('../embeddings.json', import.meta.url), payload);
console.log(`\nwrote embeddings.json (${(payload.length / 1024).toFixed(0)} KB, ${Object.keys(out).length} words)`);
