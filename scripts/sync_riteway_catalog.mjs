#!/usr/bin/env node

import fs from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import vm from "node:vm";

const DEFAULT_SOURCE = "https://ritewaylandscapeproducts.com/products-data.js";
const DEFAULT_OUTPUT = path.resolve(process.cwd(), "riteway_catalog.json");

async function readSource(source) {
  if (/^https?:\/\//i.test(source)) {
    const response = await fetch(source, {
      headers: { "user-agent": "riteway-voice-agent-catalog-sync/1.0" },
    });
    if (!response.ok) {
      throw new Error(`Catalog download failed with HTTP ${response.status}`);
    }
    return response.text();
  }
  return fs.readFile(path.resolve(source), "utf8");
}

function getSpec(product, label) {
  return product.specs?.find((spec) => spec.label === label)?.value || null;
}

function normalizeProduct(product) {
  return {
    slug: product.slug,
    name: product.name,
    category: product.subgroup || product.category,
    short_description: product.shortDescription,
    delivery_price: product.pricing?.delivery ?? null,
    pickup_price: product.pricing?.pickup ?? null,
    unit: product.pricing?.unit || "Yard",
    best_for: getSpec(product, "Best For"),
    coverage: getSpec(product, "Coverage"),
    nominal_size: getSpec(product, "Nominal Size"),
  };
}

const source = process.argv[2] || DEFAULT_SOURCE;
const output = path.resolve(process.argv[3] || DEFAULT_OUTPUT);
const sourceCode = await readSource(source);
const sandbox = { window: {} };

vm.runInNewContext(sourceCode, sandbox, {
  filename: source,
  timeout: 5_000,
});

const sourceProducts = sandbox.window.RITEWAY_PRODUCTS;
if (!Array.isArray(sourceProducts)) {
  throw new Error("The source did not define window.RITEWAY_PRODUCTS");
}

const products = sourceProducts
  .filter((product) => product?.purchasable === true && product?.pricing)
  .map(normalizeProduct);

const seen = new Set();
for (const product of products) {
  if (!product.slug || !product.name || seen.has(product.slug)) {
    throw new Error(`Invalid or duplicate catalog product: ${product.slug || product.name}`);
  }
  seen.add(product.slug);
}

const catalog = {
  source: /^https?:\/\//i.test(source) ? source : "https://ritewaylandscapeproducts.com/products-data.js",
  generated_at: new Date().toISOString(),
  source_product_count: sourceProducts.length,
  orderable_product_count: products.length,
  products,
};

await fs.writeFile(output, `${JSON.stringify(catalog, null, 2)}\n`, "utf8");
console.log(`Wrote ${products.length} Riteway products to ${output}`);
